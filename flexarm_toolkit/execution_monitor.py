# -*- coding: utf-8 -*-
"""
FlexArm 任务执行监控器 (Execution Monitor)
"""

from typing import List, Optional

try:
    from .task_manager import TaskManager, PlanAction, ActionType, ActionStatus
    from .robot_controller import RobotController
except ImportError:
    from task_manager import TaskManager, PlanAction, ActionType, ActionStatus
    from robot_controller import RobotController


class ExecutionMonitor:
    """逐步执行 PlanAction 并在检查点支持动态重规划。"""

    def __init__(
        self,
        controller: RobotController,
        task_manager: TaskManager,
        interactive_replan: bool = True,
    ):
        self.controller = controller
        self.tm = task_manager
        self.interactive_replan = bool(interactive_replan)
        self.auto_injected_target: Optional[str] = None
        self.has_replanned: bool = False

    def run_plan(self, plan: List[PlanAction]) -> bool:
        if not plan:
            return False

        working_plan = list(plan)
        idx = 0
        while idx < len(working_plan):
            act = working_plan[idx]
            act.status = ActionStatus.RUNNING
            try:
                if act.action_type in (ActionType.MOVE_TO, ActionType.CHECKPOINT):
                    if act.target_angles is not None:
                        self.controller.move_angles(act.target_angles, speed=act.speed)
                elif act.action_type == ActionType.GRIPPER_OPEN:
                    self.controller.gripper_open(speed=act.speed)
                elif act.action_type == ActionType.GRIPPER_CLOSE:
                    self.controller.gripper_close(speed=act.speed)
                act.status = ActionStatus.COMPLETED
            except Exception:
                act.status = ActionStatus.FAILED
                return False

            if (
                act.is_checkpoint
                and not self.has_replanned
                and self.auto_injected_target
            ):
                ok, updated_plan, _ = self.tm.replan_destination(
                    new_target_station=self.auto_injected_target,
                    current_action_index=idx,
                    speed=act.speed,
                )
                if not ok:
                    return False
                self.has_replanned = True
                working_plan = updated_plan

            idx += 1

        return True
