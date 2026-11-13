# -*- coding: utf-8 -*-
"""
FlexArm 轨迹代价评估器 (Plan Cost Evaluator)
计算关节空间移动距离、时间代价及全量重置 vs 航点在线重规划的代价对比。
"""

import math
from typing import Dict, Any, List, Sequence


class PlanCostEvaluator:
    """机械臂轨迹规划代价模型。"""

    def __init__(self, gripper_action_cost: float = 12.0, checkpoint_overhead: float = 5.0):
        self.gripper_action_cost = float(gripper_action_cost)
        self.checkpoint_overhead = float(checkpoint_overhead)

    @staticmethod
    def joint_distance(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != 6 or len(b) != 6:
            return 0.0
        return float(sum(abs(float(x) - float(y)) for x, y in zip(a, b)))

    @staticmethod
    def joint_max_delta(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != 6 or len(b) != 6:
            return 0.0
        return float(max(abs(float(x) - float(y)) for x, y in zip(a, b)))

    def estimate_plan_cost(
        self,
        actions: List[Any],
        start_angles: Sequence[float] = (5.0, -4.0, -7.0, -57.0, 5.0, -132.0),
    ) -> Dict[str, Any]:
        """估算给定动作序列的总角度位移、动作步数及综合代价分。"""
        cur = [float(v) for v in start_angles]
        total_deg = 0.0
        gripper_ops = 0
        move_ops = 0

        for act in actions:
            atype = getattr(act, "action_type", None)
            atype_val = atype.value if hasattr(atype, "value") else str(atype)
            if atype_val in ("MOVE_TO", "CHECKPOINT"):
                tgt = getattr(act, "target_angles", None)
                if tgt and len(tgt) == 6:
                    total_deg += self.joint_distance(cur, tgt)
                    cur = [float(v) for v in tgt]
                    move_ops += 1
                if atype_val == "CHECKPOINT":
                    total_deg += self.checkpoint_overhead
            elif atype_val in ("GRIPPER_OPEN", "GRIPPER_CLOSE"):
                gripper_ops += 1

        total_cost = round(total_deg + gripper_ops * self.gripper_action_cost, 2)
        return {
            "total_joint_deg": round(total_deg, 2),
            "move_steps": move_ops,
            "gripper_steps": gripper_ops,
            "total_cost": total_cost,
        }

    def compare_costs(
        self,
        baseline_actions: List[Any],
        replan_actions: List[Any],
        start_angles: Sequence[float],
    ) -> Dict[str, Any]:
        base = self.estimate_plan_cost(baseline_actions, start_angles=start_angles)
        rep = self.estimate_plan_cost(replan_actions, start_angles=start_angles)
        base_val = max(base["total_cost"], 1e-6)
        saving_pct = max(0.0, (base_val - rep["total_cost"]) / base_val * 100.0)
        return {
            "baseline": base,
            "replan": rep,
            "cost_saved": round(base["total_cost"] - rep["total_cost"], 2),
            "cost_improvement": f"{saving_pct:.1f}%",
        }
