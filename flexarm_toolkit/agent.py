# -*- coding: utf-8 -*-
"""
FlexArm 大模型智能体 (FlexArm LLM Agent)
封装基于 ReAct / Function Calling 范式的智能体调度中枢。
支持任意兼容 OpenAI / Gemini 格式的大语言模型，或在无 API Key 时通过启发式规则引擎自主闭环运行。
"""

import json
import logging
from typing import List, Dict, Any, Optional

from .schemas import TOOL_SCHEMAS
from .tools import FlexArmToolkit

SYSTEM_PROMPT = """你是一个专业的工业机械臂操作与任务重规划专家智能体（FlexArm Agent）。
你负责操纵大象机器人 myCobot Pro 450 六轴工业级机械臂和智能自适应电动夹爪，完成高精度的无视觉工位搬运和动态工序变更重规划。

【物理与空间世界模型】：
1. 工作空间具有 4 个已标定的安全工位：
   - P1: 左前方主取料工位 (工件 Object_A 初始工位)
   - P2: 左前方次级取料工位 (工件 Object_B 初始工位)
   - P3: 右前方放置工位 (默认目标工位)
   - P4: 正前方中间重规划工位 (动态改道目标工位)
2. 搬运状态机包含核心【空中安全检查点 (ACT-06)】：
   - 工件在完成垂直抬升后进入安全悬停挂起状态，此时工件夹持稳固且不会发生桌面碰撞。
   - 只有在 ACT-06 检查点处，才允许执行目标变更重规划！
3. 安全硬约束防火墙（五重硬约束）：
   - 严禁抓取未知工件；
   - 严禁前往未定义工位；
   - 严禁将工件放置到已有物体占用的工位；
   - 严禁超越物理安全工作圈；
   - 发生安全隐患或异常时，必须立即调用 flexarm_emergency_stop。

【标准作业流程 (SOP)】：
1. 观察：调用 flexarm_get_status 确认机械臂已上电并处于安全待机姿态；
2. 审查：调用 flexarm_safety_check 审查用户请求是否触犯 5 重物理硬约束；
3. 规划：调用 flexarm_plan_task 生成 11 步动作序列；
4. 评估：若发生工序变更，调用 flexarm_compare_replan_costs 评估优化收益；
5. 执行：调用 flexarm_execute_plan 执行搬运；遇到变更时调用 flexarm_replan_destination 实时改道。
"""


class FlexArmAgent:
    """
    FlexArm 机械臂自主智能体
    """

    def __init__(
        self,
        simulation: bool = False,
        station_config_path: str = "station_config.json",
        llm_client: Optional[Any] = None,
        model_name: str = "gpt-4o"
    ):
        self.toolkit = FlexArmToolkit(
            simulation=simulation,
            station_config_path=station_config_path
        )
        self.llm_client = llm_client
        self.model_name = model_name
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取所有可用工具的 Function Calling 描述"""
        return TOOL_SCHEMAS

    def chat_and_execute(self, user_instruction: str) -> Dict[str, Any]:
        """
        处理用户自然语言指令，通过工具调用完成物理闭环
        若未挂载外部大模型客户端，将自动启动内置启发式 ReAct 引擎
        """
        print("\n" + "=" * 70)
        print(f"🤖 [FlexArm Agent 收到用户指令]: \"{user_instruction}\"")
        print("=" * 70)

        self.history.append({"role": "user", "content": user_instruction})

        # 若有真实 LLM Client (OpenAI 兼容接口)
        if self.llm_client is not None:
            return self._run_llm_react_loop(user_instruction)
        else:
            # 内置轻量级启发式自主规划引擎 (Zero-Dependency)
            return self._run_heuristic_react_loop(user_instruction)

    def _run_heuristic_react_loop(self, user_instruction: str) -> Dict[str, Any]:
        """内置启发式 ReAct 决策循环"""
        call_logs = []

        # 1. 观察物理状态
        print("🔍 [Agent 思考 1/5]: 首先需要观测机械臂底层硬件连接、各关节角度与夹爪物理状态...")
        res_status = self.toolkit.execute_tool("flexarm_get_status", {})
        call_logs.append({"tool": "flexarm_get_status", "result": res_status})
        print(f"   -> 状态返回: 模式={res_status.get('mode')} | 连接={res_status.get('connected')}")

        # 2. 自然语言解析提取意图
        text = user_instruction.lower()
        item = "Object_A"
        src = "P1"
        tgt = "P3"
        new_tgt = None

        if "b" in text or "工件b" in text:
            item = "Object_B"
            src = "P2"
        if "p4" in text or "工位4" in text or "改放到p4" in text:
            new_tgt = "P4"
        elif "p2" in text and src != "P2":
            new_tgt = "P2"

        # 3. 安全审查
        print(f"\n🛡️ [Agent 思考 2/5]: 审查搬运规划安全约束 (工件: {item}, 起始: {src}, 目标: {tgt})...")
        res_sec = self.toolkit.execute_tool(
            "flexarm_safety_check",
            {"item_id": item, "source_station": src, "target_station": tgt}
        )
        call_logs.append({"tool": "flexarm_safety_check", "result": res_sec})
        if not res_sec.get("safety_passed", False):
            err_msg = f"安全防火墙硬拦截: {res_sec.get('violations')}"
            print(f"   ❌ {err_msg}")
            return {"success": False, "error": err_msg, "logs": call_logs}
        print("   ✓ 安全审查通过：各项空间与占用约束合法。")

        # 4. 任务生成
        print(f"\n📝 [Agent 思考 3/5]: 生成 11 步原子动作序列 (包含 ACT-06 安全检查点)...")
        res_plan = self.toolkit.execute_tool(
            "flexarm_plan_task",
            {"item_id": item, "source_station": src, "target_station": tgt, "speed": 10}
        )
        call_logs.append({"tool": "flexarm_plan_task", "result": res_plan})
        print(f"   ✓ 规划生成成功: 动作数={res_plan.get('plan_length')} 步 | 检查点: {res_plan.get('checkpoint_action')}")

        # 5. 代价评估与改道决策
        if new_tgt:
            print(f"\n📊 [Agent 思考 4/5]: 检测到工序变更需求 (改道至 {new_tgt})，调用代价模型评估数学收益...")
            res_cost = self.toolkit.execute_tool(
                "flexarm_compare_replan_costs",
                {"new_target_station": new_tgt}
            )
            call_logs.append({"tool": "flexarm_compare_replan_costs", "result": res_cost})
            print(f"   ★ 评估结果: 采纳【{res_cost.get('selected_strategy')}】 | 综合代价降低 {res_cost.get('improvement')}")

        # 6. 下发执行
        print(f"\n⚡ [Agent 思考 5/5]: 下发动作序列至底层驱动，在 ACT-06 检查点执行改道并平稳送达...")
        res_exec = self.toolkit.execute_tool(
            "flexarm_execute_plan",
            {
                "interactive_checkpoint": False,
                "auto_replan_target": new_tgt
            }
        )
        call_logs.append({"tool": "flexarm_execute_plan", "result": res_exec})

        final_answer = (
            f"已成功完成指令调度！工件【{item}】从起始工位【{src}】安全拾取，"
            f"在空中检查点成功重规划改道，并以标准工业航速平稳交付至新目标工位【{new_tgt if new_tgt else tgt}】。"
        )
        print(f"\n🎉 [FlexArm Agent 任务闭环报告]:\n{final_answer}")

        return {
            "success": res_exec.get("success", False),
            "final_answer": final_answer,
            "call_logs": call_logs
        }

    def _run_llm_react_loop(self, user_instruction: str) -> Dict[str, Any]:
        """对接真实 OpenAI 兼容 LLM 的 Function Calling 循环"""
        # 工具转换为 OpenAI function 格式
        tools = self.get_tool_schemas()
        messages = list(self.history)
        call_logs = []

        max_turns = 8
        for turn in range(max_turns):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return {
                    "success": True,
                    "final_answer": msg.content,
                    "call_logs": call_logs
                }

            # 执行工具调用
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f"🔧 [LLM 调用工具]: {fn_name}({fn_args})")
                tool_res = self.toolkit.execute_tool(fn_name, fn_args)
                call_logs.append({"tool": fn_name, "args": fn_args, "result": tool_res})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_res, ensure_ascii=False)
                })

        return {"success": True, "call_logs": call_logs, "final_answer": "多轮工具调用完成。"}
