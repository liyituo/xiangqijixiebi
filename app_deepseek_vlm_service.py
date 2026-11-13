# -*- coding: utf-8 -*-
"""
FlexArm 机械臂具身视觉控制服务 - DeepSeek V4.1-Flash 云端直连引擎
使用官方 API Key 与 deepseek-flash 多模态视觉模型，提供极致低延时的自然语言意图理解与工件/棋盘毫米级视觉定位。
"""

import os
import sys
import json
import time
import re
import requests
from typing import Optional, Dict, Any, List
import hmac
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from env_loader import get_vlm_api_key

# 配置
DEEPSEEK_API_KEY = get_vlm_api_key("VLM_API_KEY", "DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("VLM_MODEL", os.environ.get("DEEPSEEK_MODEL", "qwen3.8-flash"))
DEEPSEEK_API_URL = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

app = FastAPI(
    title="FlexArm DeepSeek-V4.1-Flash 极速具身视觉控制服务",
    description="基于 DeepSeek-V4.1-Flash (Vision) 的机械臂自然语言意图理解与工件/棋盘视觉感知 API",
    version="3.0.0"
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

TOOLS_SPEC = [
    {
        "name": "flexarm_pick_and_place",
        "description": "基于视觉定位目标工件/棋子在图像中的归一化检测框 [ymin, xmin, ymax, xmax] (0~1000)，驱动机械臂抓取并放置到目标工位",
        "parameters": {
            "type": "object",
            "properties": {
                "target_object": {"type": "string", "description": "目标物品或棋子名称"},
                "normalized_bbox": {"type": "array", "items": {"type": "integer"}, "description": "[ymin, xmin, ymax, xmax] 0~1000"},
                "target_station": {"type": "string", "description": "目标工位，默认 P3，可选 NORMAL, P1, P2, P3, P4, P5"}
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
        "description": "机械臂复位待机",
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

SYSTEM_PROMPT = f"""你是一个工业机械臂具身视觉控制中枢。
你可以识别人类自然语言指令与工作台图像，控制机械臂完成抓取、放置或下棋。
可用工具定义如下：
```json
{json.dumps(TOOLS_SPEC, ensure_ascii=False, indent=2)}
```

规则：
1. 若人类要求抓取、移动或下棋子/工件，请在图中精确定位该物体，输出包含 `flexarm_pick_and_place` 的 JSON。
2. 坐标 `normalized_bbox` 必须为 [ymin, xmin, ymax, xmax]，范围归一化至 0~1000 整数。
3. 如果只是问候、挥手或停机，调用相应工具。
4. 必须只输出合法 JSON 字典：{{"name": "工具名", "arguments": {{...}}, "description": "简要说明"}}，严禁废话。"""


def clean_json_text(text: str) -> Optional[Dict[str, Any]]:
    """从大模型输出中稳健提取 JSON 数据结构"""
    if not text:
        return None
    # 1. 尝试去除 Markdown 代码块包裹
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        for b in blocks:
            try:
                return json.loads(b.strip())
            except Exception:
                pass

    # 2. 匹配最外层花括号
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            candidate = text[first_brace:last_brace+1]
            return json.loads(candidate)
        except Exception:
            pass
    return None


class VisionCommandRequest(BaseModel):
    instruction: str = Field(..., description="控制指令")
    image_base64: Optional[str] = Field(None, description="图像 Base64 编码")


class VisionCommandResponse(BaseModel):
    success: bool
    instruction: str
    has_tool_call: bool
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = {}
    visual_description: Optional[str] = None
    elapsed_ms: float


class DetectAllRequest(BaseModel):
    image_base64: str = Field(..., description="工作台全景 Base64 编码")


class ObjectDetectionItem(BaseModel):
    name: str
    bbox: List[int]
    center_norm: List[float]


class DetectAllResponse(BaseModel):
    success: bool
    objects: List[ObjectDetectionItem] = []
    elapsed_ms: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "provider": "DeepSeek Cloud API",
        "model": DEEPSEEK_MODEL,
        "api_key_configured": bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.startswith("sk-"))
    }


@app.post("/robot_vision_command", response_model=VisionCommandResponse)
def handle_robot_vision_command(req: VisionCommandRequest):
    """处理多模态视觉指令（解析自然语言 + 定位抓取工件或下棋）"""
    t0 = time.time()
    user_content = []

    # 规范化 Base64
    if req.image_base64:
        b64 = req.image_base64
        if "base64," in b64:
            b64 = b64.split("base64,")[1]
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    user_content.append({"type": "text", "text": req.instruction})

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0.1
    }

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=25.0
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"DeepSeek API 报错: {resp.text}")

        res_json = resp.json()
        raw_reply = res_json["choices"][0]["message"].get("content", "")
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        tool_data = clean_json_text(raw_reply)
        if tool_data and ("name" in tool_data or "arguments" in tool_data):
            tool_name = tool_data.get("name", "flexarm_pick_and_place")
            args = tool_data.get("arguments", tool_data)
            # 确保 normalized_bbox 存在
            if "normalized_bbox" not in args and "bbox" in args:
                args["normalized_bbox"] = args["bbox"]
            if "target_station" not in args:
                args["target_station"] = "P3"

            return VisionCommandResponse(
                success=True,
                instruction=req.instruction,
                has_tool_call=True,
                tool_name=tool_name,
                arguments=args,
                visual_description=tool_data.get("description", raw_reply[:80]),
                elapsed_ms=elapsed_ms
            )

        return VisionCommandResponse(
            success=True,
            instruction=req.instruction,
            has_tool_call=False,
            tool_name=None,
            arguments={},
            visual_description=raw_reply.strip(),
            elapsed_ms=elapsed_ms
        )

    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        raise HTTPException(status_code=500, detail=f"DeepSeek 模型调用失败: {e}")


@app.post("/robot_command", response_model=VisionCommandResponse)
def handle_text_command(req: VisionCommandRequest):
    return handle_robot_vision_command(req)


@app.post("/detect_all_objects", response_model=DetectAllResponse)
def detect_all_objects(req: DetectAllRequest):
    """全景实体与工件/棋盘检测"""
    t0 = time.time()
    b64 = req.image_base64
    if "base64," in b64:
        b64 = b64.split("base64,")[1]

    prompt = (
        "你是工业机械臂具身视觉感知中枢。请对工作台画面中的象棋棋子与工件进行【精确目标提取与定位】：\n"
        "【核心识别规则】：\n"
        "1. 象棋棋子必须识别出阵营颜色与具体汉字，名称必须以'红'或'黑'开头，例如'红兵'、'红车'、'红炮'、'红相'、'红仕'、'红帅'、'黑卒'、'黑车'、'黑马'、'黑炮'、'黑士'、'黑象'、'黑将'。严禁含糊地只输出'棋子'！\n"
        "2. 其他工件请给出具体类别（如'胶带'、'圆筒'、'零件盒'等）。\n"
        "3. 坚决过滤剔除：桌面底板、白色托盘底、电源线缆、机械臂机身本体等非抓取作业环境背景物。\n"
        "输出严格合法 JSON 格式: {\"objects\": [{\"name\": \"红兵\", \"bbox\": [ymin, xmin, ymax, xmax]}]}\n"
        "坐标必须归一化到 0~1000 范围整数。严禁输出任何多余废话或解释。"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ],
        "max_tokens": 800,
        "temperature": 0.1
    }

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30.0
        )
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        if resp.status_code != 200:
            return DetectAllResponse(success=False, objects=[], elapsed_ms=elapsed_ms)

        res_json = resp.json()
        raw_reply = res_json["choices"][0]["message"].get("content", "")
        data = clean_json_text(raw_reply)

        items = []
        if data and isinstance(data.get("objects"), list):
            for obj in data["objects"]:
                name = obj.get("name", "工件")
                bbox = obj.get("bbox")
                if bbox and len(bbox) == 4:
                    ymin, xmin, ymax, xmax = [int(v) for v in bbox]
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    items.append(ObjectDetectionItem(
                        name=name,
                        bbox=[ymin, xmin, ymax, xmax],
                        center_norm=[cx, cy]
                    ))

        return DetectAllResponse(success=True, objects=items, elapsed_ms=elapsed_ms)

    except Exception:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        return DetectAllResponse(success=False, objects=[], elapsed_ms=elapsed_ms)


if __name__ == "__main__":
    import uvicorn
    print("=" * 65)
    print("  🚀 正在启动 FlexArm x DeepSeek-V4.1-Flash 云端视觉控制服务...")
    key_preview = (
        f"{DEEPSEEK_API_KEY[:8]}****************{DEEPSEEK_API_KEY[-4:]}"
        if len(DEEPSEEK_API_KEY) >= 12
        else ("已配置" if DEEPSEEK_API_KEY else "未配置 (请在 .env 中设置 VLM_API_KEY)")
    )
    print(f"  🔑 API Key: {key_preview}")
    print(f"  🧠 Model:   {DEEPSEEK_MODEL}")
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "6006"))
    print(f"  🌐 端点:     http://{host}:{port}")
    print("=" * 65)
    uvicorn.run(app, host=host, port=port)
