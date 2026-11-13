# -*- coding: utf-8 -*-
"""FlexArm Toolkit 离线测试套件（无真机依赖）。"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flexarm_toolkit import FlexArmToolkit, FlexArmAgent, TOOL_SCHEMAS


def test_package_imports_and_schemas():
    assert len(TOOL_SCHEMAS) >= 4
    assert FlexArmAgent is not None
    tk = FlexArmToolkit(simulation=True, auto_connect=True)
    res = tk.flexarm_get_status()
    assert res["success"] is True
    assert res["connected"] is True
    assert res["mode"] == "SIMULATION"
    assert len(res["joint_angles_deg"]) == 6
    assert res["gripper_status"]["angle"] == 15


def test_plan_and_execute_simulation():
    tk = FlexArmToolkit(simulation=True, auto_connect=True)
    plan_res = tk.flexarm_plan_task("CUBE_A", "P1", "P3", speed=20)
    assert plan_res["success"] is True
    assert plan_res["plan_length"] == 11
    assert plan_res["checkpoint_action"] == "ACT-06"

    exec_res = tk.flexarm_execute_plan(
        interactive_checkpoint=False, auto_replan_target="P4"
    )
    assert exec_res["success"] is True
    assert exec_res["has_replanned"] is True
    assert tk.tm.current_target == "P4"


def test_validator_rejects_invalid_stations_and_speed():
    tk = FlexArmToolkit(simulation=True, auto_connect=True)
    bad_station = tk.flexarm_plan_task("CUBE_A", "P1", "UNKNOWN_X")
    assert bad_station["success"] is False

    same_station = tk.flexarm_plan_task("CUBE_A", "P2", "P2")
    assert same_station["success"] is False

    bad_speed_res = tk.validator.validate_plan_json(
        {"item_id": "CUBE_A", "source_station": "P1", "target_station": "P3", "speed": 999}
    )
    assert bad_speed_res.is_valid is False


def test_replan_costs_and_emergency_stop():
    tk = FlexArmToolkit(simulation=True, auto_connect=True)
    tk.flexarm_plan_task("CUBE_A", "P1", "P3", speed=20)
    comp = tk.flexarm_compare_replan_costs("P4")
    assert comp["success"] is True
    assert comp["is_valid"] is True
    assert comp["selected_strategy"] == "incremental_checkpoint_suffix"

    stop_res = tk.flexarm_emergency_stop("test_estop")
    assert stop_res["success"] is True
