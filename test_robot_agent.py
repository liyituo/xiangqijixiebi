import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/autodl-tmp/Qwen2.5-1.5B-Instruct"

print("正在将模型加载到 RTX 4080 GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print("模型加载完成！显存占用仅约 3.5GB。")

# 定义机械臂当前支持的 3 个核心工具函数
tools = [
    {
        "type": "function",
        "function": {
            "name": "reset_arm",
            "description": "机械臂复位到安全初始姿态",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wave_hand",
            "description": "控制机械臂挥手或打招呼",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pick_and_place",
            "description": "从指定起点抓取物体并搬运放置到目标终点",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "抓取起始位置或物品名称，例如 'table_A'"},
                    "target": {"type": "string", "description": "放置目标位置，例如 'box_B'"}
                },
                "required": ["source", "target"]
            }
        }
    }
]

def parse_robot_command(user_instruction: str):
    """输入自然语言，大模型理解并输出机械臂工具调用"""
    messages = [
        {"role": "system", "content": "你是一个能够控制机械臂执行物理任务的智能助手。请严格根据用户指令调用对应的工具函数。"},
        {"role": "user", "content": user_instruction}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1  # 机械臂控制要求确定性，使用低采样温度
        )
    elapsed = time.time() - start_time
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    return response.strip(), elapsed

if __name__ == "__main__":
    test_cases = [
        "你好机械臂，见到你很高兴，跟我打个招呼吧！",
        "帮我把工位A上的螺母搬运到收纳盒B里面",
        "任务完成了，回到初始姿态复位休息吧"
    ]
    
    print("\n" + "="*60)
    print("开始测试【大模型自然语言 -> 机械臂工具调用】")
    print("="*60)
    
    for instruction in test_cases:
        call_result, cost_time = parse_robot_command(instruction)
        print(f"\n【用户指令】: {instruction}")
        print(f"【推理耗时】: {cost_time*1000:.1f} ms")
        print(f"【模型输出】:\n{call_result}")
        print("-" * 60)
