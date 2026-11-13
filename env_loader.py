# -*- coding: utf-8 -*-
"""
本地环境变量 (.env) 自动加载器。
避免在源码中硬编码 API Key，同时确保本机通过 .env 开箱即用。
"""

import os
from typing import Optional

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_local_env(env_path: Optional[str] = None) -> None:
    """从工作区 .env 文件加载未设置的环境变量"""
    target = env_path or _ENV_PATH
    if not os.path.exists(target):
        return
    try:
        with open(target, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"⚠️ 读取本地 .env 配置失败: {e}")


def get_vlm_api_key(*var_names: str, default: str = "") -> str:
    """加载 .env 后按优先级读取 API Key（不硬编码默认密钥）"""
    load_local_env()
    names = var_names or ("VLM_API_KEY", "DEEPSEEK_API_KEY")
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default
