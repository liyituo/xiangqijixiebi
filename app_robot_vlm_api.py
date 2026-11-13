import os
import sys
import json
import time
import re
import torch
import base64
from io import BytesIO
import hmac
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI(
    title="FlexArm 多模态视觉大模型极速控制服务",
    description="极致低延时优化的 Qwen2.5-VL-7B 机械臂视觉控制 API",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://127.0.0.1:5000,http://localhost:5000").split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Robot-Token"],
)


@app.middleware("http")
async def _verify_api_token(request: Request, call_next):
    expected = os.environ.get("ROBOT_API_TOKEN", "").strip()
    if expected and request.url.path not in ("/health", "/docs", "/openapi.json"):
        auth_hdr = request.headers.get("Authorization", "")
        token = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else request.headers.get("X-Robot-Token", "").strip()
        if not token or not hmac.compare_digest(token, expected):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)

MODEL_PATH = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"

print("==================================================")
print("正在加载极速多模态视觉引擎 (开启 SDPA + 视觉 Token 压缩)...")
print("==================================================")

# 高清模式: 提升视觉 Token 上限至 1280 像素级别，确保棋盘网格、文字细节（車馬炮）清晰可辨
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    attn_implementation="sdpa",
    device_map="cuda"
)
print("✅ Qwen2.5-VL 极速推理引擎加载完成！已就绪。")

TOOLS = [
    {
        "name": "flexarm_pick_and_place",
        "description": "基于视觉定位目标工件在图像中的归一化检测框 [ymin, xmin, ymax, xmax]，驱动机械臂抓取并放置到目标工位",
        "parameters": {
            "type": "object",
            "properties": {
                "target_object": {"type": "string", "description": "目标物品名称"},
                "normalized_bbox": {"type": "array", "items": {"type": "integer"}, "description": "[ymin, xmin, ymax, xmax] 0~1000"},
                "target_station": {"type": "string", "description": "目标工位，支持 NORMAL, P1, P2, P3, P4, P5"}
            },
            "required": ["target_object", "normalized_bbox", "target_station"]
        }
    },
    {
        "name": "flexarm_control_gripper",
        "description": "控制夹爪开合",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["OPEN", "CLOSE"]}},
            "required": ["action"]
        }
    },
    {
        "name": "reset_arm",
        "description": "机械臂复位",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "wave_hand",
        "description": "机械臂挥手打招呼",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "flexarm_emergency_stop",
        "description": "急停",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}
    }
]

SYSTEM_PROMPT = f"""你是一个工业机械臂视觉中枢。
你可以调用以下工具函数控制机械臂：
```json
{json.dumps(TOOLS, ensure_ascii=False, indent=2)}
```
请观察图像与指令。你必须【严格且仅输出】如下格式的工具 JSON 字典，严禁输出任何废话、解释或伪代码：
{{"name": "工具名", "arguments": {{参数}}}}"""

def extract_json_or_tool(text: str) -> Optional[Dict[str, Any]]:
    """鲁棒提取 JSON 工具调用"""
    # 1. 尝试 tool_call 标签
    tool_match = re.search(r"<tool_call>[\s\r\n]*({[\s\S]*?})[\s\r\n]*</tool_call>", text)
    if tool_match:
        try:
            return json.loads(tool_match.group(1))
        except Exception:
            pass

    # 2. 匹配最外层花括号 (贪婪跨多层嵌套)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            candidate = text[first_brace:last_brace+1]
            data = json.loads(candidate)
            if "name" in data:
                return data
        except Exception:
            pass
    return None

class VisionCommandRequest(BaseModel):
    instruction: str = Field(..., description="指令")
    image_base64: Optional[str] = Field(None, description="图像 Base64")

class VisionCommandResponse(BaseModel):
    success: bool
    instruction: str
    has_tool_call: bool
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = {}
    visual_description: Optional[str] = None
    elapsed_ms: float

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Qwen2.5-VL-7B-Instruct (Turbo)",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "vram_gb": round(torch.cuda.memory_allocated(0)/(1024**3), 2) if torch.cuda.is_available() else 0
    }

@app.post("/robot_vision_command", response_model=VisionCommandResponse)
def handle_vision_command(req: VisionCommandRequest):
    t0 = time.time()
    
    content_list = []
    if req.image_base64:
        b64_data = req.image_base64
        if "base64," in b64_data:
            b64_data = b64_data.split("base64,")[1]
        try:
            img = Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")
            # 保持高清原生比例，最大边支持到 1280，保留棋盘与棋子字迹细节
            img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            content_list.append({"type": "image", "image": img})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图片解码失败: {e}")
            
    content_list.append({"type": "text", "text": req.instruction})
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_list}
    ]
    
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.01,
            do_sample=False
        )
        
    elapsed_ms = (time.time() - t0) * 1000
    gen_text = processor.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    
    tool_data = extract_json_or_tool(gen_text)
    if tool_data and "name" in tool_data:
        return VisionCommandResponse(
            success=True,
            instruction=req.instruction,
            has_tool_call=True,
            tool_name=tool_data.get("name"),
            arguments=tool_data.get("arguments", {}),
            visual_description=None,
            elapsed_ms=round(elapsed_ms, 2)
        )

    return VisionCommandResponse(
        success=True,
        instruction=req.instruction,
        has_tool_call=False,
        tool_name=None,
        arguments={},
        visual_description=gen_text.replace("<|im_end|>", "").strip(),
        elapsed_ms=round(elapsed_ms, 2)
    )

@app.post("/robot_command", response_model=VisionCommandResponse)
def handle_text_command(req: VisionCommandRequest):
    return handle_vision_command(req)

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "6006"))
    uvicorn.run(app, host=host, port=port)
