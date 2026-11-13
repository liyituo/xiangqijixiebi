import os
import sys
import json
import time
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(
    title="FlexArm 大模型具身控制服务",
    description="基于 Qwen2.5-1.5B-Instruct 与 NVIDIA RTX 4080 的机械臂自然语言意图控制 API",
    version="1.0.0"
)

MODEL_PATH = "/root/autodl-tmp/Qwen2.5-1.5B-Instruct"

print(f"正在加载大模型: {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print("大模型加载完成！服务准备就绪。")

# 机械臂全套支持工具列表 (涵盖 FlexArm 工具集 + 基础关节与动作工具)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "flexarm_get_status",
            "description": "获取机械臂当前完整物理状态（各关节角度J1-J6、末端坐标XYZ、夹爪状态与工位状态）",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_safety_check",
            "description": "审查搬运任务是否违反物理安全硬约束（工位白名单、工件合法性、碰撞防干涉）",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "工件标识符 (如 'Object_A', 'Object_B')"},
                    "source_station": {"type": "string", "description": "起始取料工位 (P1~P4)"},
                    "target_station": {"type": "string", "description": "目标下放工位 (P1~P4)"}
                },
                "required": ["item_id", "source_station", "target_station"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_plan_task",
            "description": "生成标准搬运任务动作序列（含 ACT-06 空中安全检查点）",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "被搬运工件标识符"},
                    "source_station": {"type": "string", "description": "起始工位 (如 'P1')"},
                    "target_station": {"type": "string", "description": "目标工位 (如 'P3')"},
                    "speed": {"type": "integer", "description": "机械臂运行速度(5~20)", "default": 10}
                },
                "required": ["item_id", "source_station", "target_station"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_execute_plan",
            "description": "驱动机械臂执行生成的动作规划",
            "parameters": {
                "type": "object",
                "properties": {
                    "auto_replan_target": {"type": "string", "description": "若需改道，指定新的目标工位(如 'P4')"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_control_gripper",
            "description": "控制机械臂末端自适应电动夹爪打开或闭合",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["OPEN", "CLOSE"], "description": "夹爪动作: OPEN(张开) 或 CLOSE(夹紧)"},
                    "speed": {"type": "integer", "description": "开闭速度(1~100)", "default": 50}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_emergency_stop",
            "description": "【紧急停止】立即切断机械臂电机使能并紧急刹停",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "急停原因"}
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wave_hand",
            "description": "控制机械臂向用户打招呼或摆动",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reset_arm",
            "description": "控制机械臂回到安全初始待机姿态",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

SYSTEM_PROMPT = """你是一个专业的工业机械臂操作智能体。
你负责操纵大象机器人 6 轴机械臂与电动夹爪。
当用户输入自然语言指令时，请依据用户意图，从所提供的工具列表中调用最合适的工具函数。
如果没有对应的具体动作，请直接友好回复用户。"""

class CommandRequest(BaseModel):
    instruction: str = Field(..., description="用户的自然语言指令，如'把A从P1搬运到P3'或'打个招呼'")

class CommandResponse(BaseModel):
    success: bool
    instruction: str
    has_tool_call: bool
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = {}
    reply_text: Optional[str] = None
    elapsed_ms: float

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model": "Qwen2.5-1.5B-Instruct",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated(0)/(1024**3), 2) if torch.cuda.is_available() else 0,
        "supported_tools_count": len(TOOLS)
    }

@app.post("/robot_command", response_model=CommandResponse)
def handle_robot_command(req: CommandRequest):
    """
    核心接口：接收自然语言指令，大模型推理并输出结构化机械臂工具调用
    """
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.instruction}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    gen_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    
    # 解析模型输出中的 tool_call
    # 格式通常形如: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
    tool_pattern = r"<tool_call>[\s\r\n]*({.*?})[\s\r\n]*</tool_call>"
    match = re.search(tool_pattern, gen_text, re.DOTALL)
    
    if match:
        try:
            tool_data = json.loads(match.group(1))
            tool_name = tool_data.get("name")
            arguments = tool_data.get("arguments", {})
            return CommandResponse(
                success=True,
                instruction=req.instruction,
                has_tool_call=True,
                tool_name=tool_name,
                arguments=arguments,
                reply_text=gen_text.replace(match.group(0), "").replace("<|im_end|>", "").strip(),
                elapsed_ms=round(elapsed_ms, 2)
            )
        except Exception as e:
            pass
            
    # 如果没有结构化 tool_call 或者仅纯文本回复
    cleaned_reply = gen_text.replace("<|im_end|>", "").strip()
    return CommandResponse(
        success=True,
        instruction=req.instruction,
        has_tool_call=False,
        tool_name=None,
        arguments={},
        reply_text=cleaned_reply,
        elapsed_ms=round(elapsed_ms, 2)
    )

if __name__ == "__main__":
    import uvicorn
    # 监听 0.0.0.0:6006 (AutoDL 标准服务端口)
    uvicorn.run(app, host="0.0.0.0", port=6006)
