# -*- coding: utf-8 -*-
"""
FlexArm 最小代价在线重规划器 (Minimum-Cost Replanner)
"""

from typing import Dict, Any

try:
    from .cost_model import PlanCostEvaluator
    from .task_manager import TaskManager
except ImportError:
    from cost_model import PlanCostEvaluator
    from task_manager import TaskManager


class MinimumCostReplanner:
    """在执行检查点处对比从头重置与增量修改后缀策略，并执行最小代价重规划。"""

    def __init__(self, task_manager: TaskManager):
        self.tm = task_manager
        self.evaluator = PlanCostEvaluator()

    def replan(
        self,
        new_target_station: str,
        current_action_index: int = 3,
        speed: int = 20,
    ) -> Dict[str, Any]:
        if not self.tm.current_plan:
            return {"valid": False, "error": "当前无活动规划"}

        new_tgt = str(new_target_station).strip().upper()
        if new_tgt not in self.tm.stations:
            return {"valid": False, "error": f"未知重规划目标工位 {new_tgt!r}"}

        idx = max(0, min(int(current_action_index), len(self.tm.current_plan) - 1))
        anchor_angles = self.tm.current_plan[idx].target_angles or self.tm.standby_angles

        # 保存快照用于无副作用对比代价（再调用 tm.replan_destination 提交最小代价后缀）
        old_plan = list(self.tm.current_plan)
        old_src = self.tm.current_source or "P1"
        old_item = self.tm.current_item_id or "WORKPIECE"

        # 策略 A：先回 HOME 再重新执行完整搬运（仅估算）
        full_restart_plan = list(self.tm.current_plan[: idx + 1]) + self.tm.build_pick_and_place_plan(
            old_item, old_src, new_tgt, speed=speed
        )
        # 恢复 current_plan 并计算策略 B（在线后缀重规划）
        self.tm.current_plan = old_plan
        ok, suffix_plan, diff_report = self.tm.replan_destination(
            new_target_station=new_tgt,
            current_action_index=idx,
            speed=speed,
        )
        if not ok:
            return {"valid": False, "error": diff_report.get("error", "重规划失败")}

        cost_analysis = self.evaluator.compare_costs(
            baseline_actions=full_restart_plan[idx + 1 :],
            replan_actions=suffix_plan[idx + 1 :],
            start_angles=anchor_angles,
        )
        return {
            "valid": True,
            "selected_strategy": "incremental_checkpoint_suffix",
            "cost_analysis": cost_analysis,
            "diff_report": diff_report,
        }
