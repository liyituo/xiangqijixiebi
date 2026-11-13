# -*- coding: utf-8 -*-
"""
FlexArm 任务 JSON 指令安全与合法性校验器。
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class InstructionValidator:
    """校验大模型下发的搬运任务参数是否满足工位与物理约束。"""

    def __init__(self, station_config_path: str):
        with open(station_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.stations = {str(k).upper() for k in cfg.get("stations", {}).keys()}

    def _validate_single_task(self, task: Dict[str, Any], prefix: str = "") -> List[str]:
        errs: List[str] = []
        item_id = task.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            errs.append(f"{prefix}缺少有效 item_id")

        src_raw = task.get("from_station", task.get("source_station", ""))
        tgt_raw = task.get("to_station", task.get("target_station", ""))
        src = str(src_raw or "").strip().upper()
        tgt = str(tgt_raw or "").strip().upper()

        if not src or src not in self.stations:
            errs.append(f"{prefix}未知源工位 source_station={src!r}")
        if not tgt or tgt not in self.stations:
            errs.append(f"{prefix}未知目标工位 target_station={tgt!r}")
        if src and tgt and src == tgt:
            errs.append(f"{prefix}源工位与目标工位相同 ({src})，拒绝无效搬运")

        if "speed" in task and task.get("speed") is not None:
            speed = task.get("speed")
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(float(speed)):
                errs.append(f"{prefix}非法速度值 speed={speed!r}")
            else:
                s_val = float(speed)
                if s_val < 5 or s_val > 80:
                    errs.append(f"{prefix}速度 speed={s_val} 超出安全范围 [5, 80]")
        return errs

    def validate_plan_json(self, task_dict: Dict[str, Any]) -> ValidationResult:
        if not isinstance(task_dict, dict):
            return ValidationResult(False, ["任务载荷必须为字典对象"])

        errors: List[str] = []
        tasks = task_dict.get("tasks")
        if isinstance(tasks, list):
            if not tasks:
                errors.append("tasks 列表为空")
            else:
                for i, t in enumerate(tasks):
                    if not isinstance(t, dict):
                        errors.append(f"tasks[{i}] 必须为字典")
                    else:
                        errors.extend(self._validate_single_task(t, prefix=f"tasks[{i}]: "))
        else:
            errors.extend(self._validate_single_task(task_dict))

        return ValidationResult(is_valid=(len(errors) == 0), errors=errors)
