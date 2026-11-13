# -*- coding: utf-8 -*-
"""
FlexArm x OpenVLA-7B 具身智能动作生成服务 (OpenVLA Serving API)
基于 RTX 4090 GPU (24GB VRAM) 提供极速视觉-语言-动作 (VLA) 推理服务：
1. 接收工业相机拍摄的最新 RGB 图像与自然语言任务指令
2. 经由 Prismatic VLM + LLaMA-2-7B 骨干网前向推理
3. 实时输出 7 自由度机械臂末端动作增量 [dx, dy, dz, droll, dpitch, dyaw, gripper]
4. 兼具原生 predict_action 接口与统一 robot_vision_command 兼容接口
"""

import os
import sys
import time
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# 添加模型目录至系统路径
MODEL_PATH = "/root/autodl-tmp/openvla-7b"
if os.path.exists(MODEL_PATH) and MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)

from transformers import AutoModelForVision2Seq, AutoProcessor

app = FastAPI(
    title="FlexArm OpenVLA-7B Embodied Serving API",
    description="基于 OpenVLA-7B 与 RTX 4090 的端到端具身操作动作生成服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型与处理器单例
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_openvla():
    global processor, model
    print(f"🚀 [INIT] 正在加载 OpenVLA-7B 具身大模型至 {device} (RTX 4090)...")
    t0 = time.time()
    
    # 1. 加载 Processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"✅ Processor 加载就绪 (耗时: {time.time()-t0:.2f}s)")
    
    # 2. 加载 Model (bfloat16 显存优化)
    t1 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    alloc_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"✅ OpenVLA-7B 权重载入完成！显存占用: {alloc_gb:.2f} GB (耗时: {time.time()-t1:.2f}s)")

# ------------------------------------------------------------------------------
# 数据结构模型
# ------------------------------------------------------------------------------

class PredictActionRequest(BaseModel):
    instruction: str = Field(..., description="机械臂自然语言操作任务描述")
    image_base64: str = Field(..., description="工作台摄像头的当前帧 Base64 编码 (RGB)")
    unnorm_key: Optional[str] = Field("bridge_orig", description="数据集动作反归一化 Key")
    do_sample: bool = Field(False, description="是否采样 (False 为贪婪确定性输出)")

class VisionCommandRequest(BaseModel):
    instruction: str = Field(..., description="用户语音或自然语言指令")
    image_base64: str = Field(..., description="摄像头捕获的图像 Base64")

def decode_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

# ------------------------------------------------------------------------------
# RESTful API 端点
# ------------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    try:
        load_openvla()
    except Exception as e:
        print(f"⚠️ 启动阶段模型加载延后: {e}")

@app.get("/health")
def health_check():
    """服务健康检查接口"""
    vram_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    vram_res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    return {
        "status": "healthy" if model is not None else "model_loading",
        "model_name": "openvla/openvla-7b",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_allocated_gb": round(vram_alloc, 2),
        "vram_reserved_gb": round(vram_res, 2),
        "timestamp": time.time()
    }

@app.post("/predict_action")
def predict_action_endpoint(req: PredictActionRequest):
    """
    OpenVLA 原生动作预测接口：
    输入: 指令文本 + 当前图像
    输出: 7-DoF 连续动作增量 [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="OpenVLA 模型未加载完成，请稍候重试")

    t0 = time.time()
    try:
        img = decode_image(req.image_base64)
        prompt = f"In: What action should the robot take to {req.instruction}?\nOut:"
        
        inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = model.predict_action(
                **inputs,
                unnorm_key=req.unnorm_key,
                do_sample=req.do_sample
            )

        elapsed_ms = (time.time() - t0) * 1000.0
        action_list = [float(x) for x in action.tolist()]

        return {
            "success": True,
            "instruction": req.instruction,
            "action_7d": action_list,
            "elapsed_ms": round(elapsed_ms, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"动作预测异常: {str(e)}")

@app.post("/robot_vision_command")
def robot_vision_command_endpoint(req: VisionCommandRequest):
    """
    与本地 Web 大屏 / 视觉客户端完全兼容的通用具身调度接口：
    将 OpenVLA 预测的动作直接包装为标准 tool_call 供客户端执行
    """
    if model is None or processor is None:
        return {
            "has_tool_call": False,
            "visual_description": "云端 OpenVLA-7B 模型正在装载中，请稍候...",
            "elapsed_ms": 0.0
        }

    t0 = time.time()
    try:
        img = decode_image(req.image_base64)
        prompt = f"In: What action should the robot take to {req.instruction}?\nOut:"
        
        inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False
            )

        elapsed_ms = (time.time() - t0) * 1000.0
        action_list = [float(x) for x in action.tolist()]

        return {
            "has_tool_call": True,
            "tool_name": "openvla_step",
            "arguments": {
                "action_7d": action_list,
                "instruction": req.instruction
            },
            "elapsed_ms": round(elapsed_ms, 1),
            "visual_description": f"OpenVLA 预测动作: dXYZ=[{action_list[0]:.2f}, {action_list[1]:.2f}, {action_list[2]:.2f}] 夹爪={action_list[6]:.2f}"
        }
    except Exception as e:
        return {
            "has_tool_call": False,
            "error": str(e),
            "visual_description": f"推理发生异常: {e}",
            "elapsed_ms": (time.time() - t0) * 1000.0
        }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  🚀 启动 OpenVLA-7B 具身控制服务 (Port: 6006)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=6006, log_level="info")
