# -*- coding: utf-8 -*-
"""
FlexArm 机械臂大模型工具包 (FlexArm LLM Toolkit)
面向动态工序变更与最小代价重规划的机械臂智能体工具包。

导出组件：
- FlexArmToolkit: 统一硬件控制与重规划工具类
- FlexArmAgent: 自主 ReAct / Function Calling 智能体
- TOOL_SCHEMAS: 符合 OpenAI / Gemini / Anthropic 标准的 8 大工具定义
- SYSTEM_PROMPT: 专为工业机械臂操作优化的专业领域知识系统提示词
"""

from .schemas import TOOL_SCHEMAS
from .tools import FlexArmToolkit
from .agent import FlexArmAgent, SYSTEM_PROMPT

__version__ = "1.0.0"
__all__ = [
    "FlexArmToolkit",
    "FlexArmAgent",
    "TOOL_SCHEMAS",
    "SYSTEM_PROMPT"
]
