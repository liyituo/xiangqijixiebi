# -*- coding: utf-8 -*-
"""
FlexArm 大模型工具包 - 工具定义 Schema (OpenAI / Gemini / Anthropic 标准规范)
定义了供大模型智能体进行 Function Calling 的 8 项标准化工具定义。
"""

from typing import List, Dict, Any

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "flexarm_get_status",
            "description": "获取 FlexArm 机械臂当前完整物理状态。包括各关节角度 (J1~J6)、末端笛卡尔坐标 (X,Y,Z,Rx,Ry,Rz)、夹爪开合状态与力控反馈、当前各工位占用情况及硬件连接模式。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_safety_check",
            "description": "通过 5 重物理安全防火墙审查搬运任务指令。审查项目：1. 工件合法性 2. 工位白名单 3. 工作空间物理限位 4. 依赖死锁 5. 工位占用干涉冲突。任何违规将被硬拦截，严禁盲动。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "工件标识符 (如 'Object_A' 或 'Object_B')"
                    },
                    "source_station": {
                        "type": "string",
                        "description": "起始取料工位 (如 'P1' 或 'P2')"
                    },
                    "target_station": {
                        "type": "string",
                        "description": "目标下放工位 (如 'P3' 或 'P4')"
                    }
                },
                "required": ["item_id", "source_station", "target_station"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_plan_task",
            "description": "生成标准无视觉搬运的 11 步原子动作序列 (Action Primitives)。包含正向抓取、垂直抬升、空中安全检查点 (ACT-06)、空间平移、垂直下放与返回待机。默认航速 10。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "被搬运的工件标识符"
                    },
                    "source_station": {
                        "type": "string",
                        "description": "起始抓取工位编号 (P1 ~ P4)"
                    },
                    "target_station": {
                        "type": "string",
                        "description": "目标放置工位编号 (P1 ~ P4)"
                    },
                    "speed": {
                        "type": "integer",
                        "description": "标准工业运动速度 (推荐 10，安全范围 5~20)",
                        "default": 10
                    }
                },
                "required": ["item_id", "source_station", "target_station"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_execute_plan",
            "description": "启动执行当前生成的动作规划。支持在空中安全检查点 (ACT-06) 自动挂起暂停，以便大模型决策是否需要改道或继续下放。",
            "parameters": {
                "type": "object",
                "properties": {
                    "interactive_checkpoint": {
                        "type": "boolean",
                        "description": "是否在空中安全检查点处静止挂起，等待确认或重规划指令 (默认 True)",
                        "default": True
                    },
                    "auto_replan_target": {
                        "type": "string",
                        "description": "可选：到达检查点后自动直接改道至该新工位 (如 'P4')，若不提供则挂起等待用户/智能体交互"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_replan_destination",
            "description": "【核心重规划接口】当工件在空中安全检查点 (ACT-06) 悬停挂起时，动态变更目标放置工位。基于 Plan Diff 算法剪除旧路径动作并无缝注入新工位平移与垂直下放动作，确保航速 >= 10，零多余回退代价。",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_target_station": {
                        "type": "string",
                        "description": "变更后的新目标工位编号 (如 'P4' 或 'P2')"
                    },
                    "speed": {
                        "type": "integer",
                        "description": "重规划平移航速 (推荐 10，不得低于 5)",
                        "default": 10
                    }
                },
                "required": ["new_target_station"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_compare_replan_costs",
            "description": "计算并对比动态工序变更下的综合代价模型。输出全量重构 (Baseline) 与 FlexArm 最小代价补丁方案的剩余动作数、回退补偿数、预估耗时及综合加权代价 (Cost)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_target_station": {
                        "type": "string",
                        "description": "拟变更的目标工位编号"
                    }
                },
                "required": ["new_target_station"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_control_gripper",
            "description": "底层直接驱动智能自适应电动夹爪进行张开或闭合动作，带有加电使能和 100% 满额力矩咬合闭环反馈。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["OPEN", "CLOSE"],
                        "description": "夹爪动作指令：'OPEN' 张开释放 / 'CLOSE' 闭合夹紧"
                    },
                    "speed": {
                        "type": "integer",
                        "description": "夹爪运行速度 (1~100，默认 100 满速)",
                        "default": 100
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flexarm_emergency_stop",
            "description": "【最高优先级安全熔断】执行紧急制动停止机械臂运动并断开连接，适用于检测到人员侵入、物理干涉或非法越界行为。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "触发紧急制动的原因说明"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]
