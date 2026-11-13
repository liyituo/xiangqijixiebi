# -*- coding: utf-8 -*-
"""
FlexArm 任务编排器与动作序列数据结构。
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class ActionType(str, Enum):
    MOVE_TO = "MOVE_TO"
    GRIPPER_OPEN = "GRIPPER_OPEN"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    CHECKPOINT = "CHECKPOINT"
    WAIT = "WAIT"


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanAction:
    action_id: str
    action_type: ActionType
    target_station: Optional[str] = None
    target_angles: Optional[List[float]] = None
    speed: int = 20
    is_checkpoint: bool = False
    description: str = ""
    status: ActionStatus = ActionStatus.PENDING


class TaskManager:
    """管理站点配置、拾取-搬运 11 步任务规划与在线目标重规划。"""

    def __init__(self, station_config_path: str):
        self.station_config_path = station_config_path
        with open(station_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.raw_config: Dict[str, Any] = cfg
        self.stations: Dict[str, Dict[str, Any]] = {
            str(k).upper(): v for k, v in cfg.get("stations", {}).items()
        }
        standby_key = str(cfg.get("standby_station", "NORMAL")).upper()
        self.standby_angles: List[float] = list(
            self.stations.get(standby_key, {}).get(
                "angles", [5.0, -4.0, -7.0, -57.0, 5.0, -132.0]
            )
        )
        self.current_plan: List[PlanAction] = []
        self.current_source: Optional[str] = None
        self.current_target: Optional[str] = None
        self.current_item_id: Optional[str] = None

    def get_station_angles(self, station_id: str) -> List[float]:
        key = str(station_id).strip().upper()
        if key not in self.stations:
            raise ValueError(f"未知工位 {station_id!r}，可选: {sorted(self.stations.keys())}")
        return [float(v) for v in self.stations[key]["angles"]]

    def build_pick_and_place_plan(
        self,
        item_id: str,
        source_station: str,
        target_station: str,
        speed: int = 20,
    ) -> List[PlanAction]:
        src = str(source_station).strip().upper()
        tgt = str(target_station).strip().upper()
        src_angles = self.get_station_angles(src)
        tgt_angles = self.get_station_angles(tgt)
        home_angles = self.get_station_angles("HOME") if "HOME" in self.stations else list(self.standby_angles)
        spd = max(5, min(80, int(speed)))

        plan = [
            PlanAction("ACT-01", ActionType.MOVE_TO, "NORMAL", list(self.standby_angles), spd, False, "安全观测/过渡就绪"),
            PlanAction("ACT-02", ActionType.GRIPPER_OPEN, speed=50, description=f"预张开夹爪准备抓取 [{item_id}]"),
            PlanAction("ACT-03", ActionType.MOVE_TO, src, src_angles, spd, False, f"悬停并抵达源工位 [{src}]"),
            PlanAction("ACT-04", ActionType.GRIPPER_CLOSE, speed=50, description=f"夹紧目标工件 [{item_id}]"),
            PlanAction("ACT-05", ActionType.WAIT, speed=10, description="夹持稳定等待确认"),
            PlanAction("ACT-06", ActionType.CHECKPOINT, "NORMAL", list(self.standby_angles), spd, True, "空中安全中继检查点（可动态重规划）"),
            PlanAction("ACT-07", ActionType.MOVE_TO, tgt, tgt_angles, spd, False, f"安全平移至目标工位 [{tgt}] 上方"),
            PlanAction("ACT-08", ActionType.WAIT, speed=10, description="目标位姿稳定等待"),
            PlanAction("ACT-09", ActionType.GRIPPER_OPEN, speed=50, description=f"在 [{tgt}] 释放工件"),
            PlanAction("ACT-10", ActionType.WAIT, speed=10, description="释放退让稳定确认"),
            PlanAction("ACT-11", ActionType.MOVE_TO, "HOME", home_angles, spd, False, "安全回归待机位 HOME"),
        ]
        self.current_plan = plan
        self.current_source = src
        self.current_target = tgt
        self.current_item_id = item_id
        return plan

    def replan_destination(
        self,
        new_target_station: str,
        current_action_index: int = 5,
        speed: int = 20,
    ) -> Tuple[bool, List[PlanAction], Dict[str, Any]]:
        if not self.current_plan:
            return False, [], {"error": "当前无活动规划"}

        new_tgt = str(new_target_station).strip().upper()
        try:
            new_tgt_angles = self.get_station_angles(new_tgt)
        except ValueError as exc:
            return False, list(self.current_plan), {"error": str(exc)}

        idx = max(0, min(int(current_action_index), len(self.current_plan) - 1))
        old_target = self.current_target
        home_angles = self.get_station_angles("HOME") if "HOME" in self.stations else list(self.standby_angles)
        spd = max(5, min(80, int(speed)))

        prefix = list(self.current_plan[: idx + 1])
        item_label = self.current_item_id or "WORKPIECE"
        suffix = [
            PlanAction("ACT-07", ActionType.MOVE_TO, new_tgt, new_tgt_angles, spd, False, f"在线重规划：搬运 [{item_label}] 至 [{new_tgt}]"),
            PlanAction("ACT-08", ActionType.WAIT, speed=10, description="重规划目标稳定等待"),
            PlanAction("ACT-09", ActionType.GRIPPER_OPEN, speed=50, description=f"在重规划工位 [{new_tgt}] 释放工件"),
            PlanAction("ACT-10", ActionType.WAIT, speed=10, description="释放退让确认"),
            PlanAction("ACT-11", ActionType.MOVE_TO, "HOME", home_angles, spd, False, "安全回归待机位 HOME"),
        ]

        updated_plan = prefix + suffix
        self.current_plan = updated_plan
        self.current_target = new_tgt

        diff_report = {
            "old_target": old_target,
            "new_target": new_tgt,
            "replan_at_index": idx,
            "preserved_steps": len(prefix),
            "replaced_steps": len(suffix),
        }
        return True, updated_plan, diff_report
