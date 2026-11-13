# -*- coding: utf-8 -*-
"""
FlexArm Toolkit 机械臂硬件/仿真统一控制器。
复用根目录 pro450_driver 安全校验、限位与急停状态机制，提供严格的通信与安全异常语义。
"""

import os
import sys
from typing import Dict, Any, List, Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from pro450_driver import (
    MyCobotPro450Driver,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_PORT,
    PRO450_LIMITS,
    is_valid_vector,
    validate_speed,
    validate_vector,
    clamp_angles,
)


class HardwareCommunicationError(RuntimeError):
    """底层机械臂通信连接或指令发送失败异常。"""


class SafetyViolationError(ValueError):
    """机械臂目标位姿、速度或安全状态违规异常。"""


class RobotController:
    """支持仿真模式与真机驱动模式的统一机械臂控制器。"""

    GRIPPER_OPEN_PERCENT = 15
    GRIPPER_CLOSE_PERCENT = 5

    def __init__(
        self,
        simulation: bool = False,
        require_human_confirmation: bool = False,
        ip: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.simulation = bool(simulation)
        self.require_human_confirmation = bool(require_human_confirmation)
        self.ip = ip or DEFAULT_ROBOT_IP
        self.port = int(port or DEFAULT_ROBOT_PORT)
        self._driver: Optional[MyCobotPro450Driver] = None
        self._connected: bool = False
        self._stopped: bool = False

        # 仿真状态初值（与 NORMAL 位姿一致）
        self._sim_angles: List[float] = [5.0, -4.0, -7.0, -57.0, 5.0, -132.0]
        self._sim_coords: List[float] = [200.0, 0.0, 260.0, 180.0, 0.0, 0.0]
        self._sim_gripper: int = self.GRIPPER_OPEN_PERCENT

    def connect(self) -> bool:
        self._stopped = False
        if self.simulation:
            self._connected = True
            return True
        try:
            self._driver = MyCobotPro450Driver(
                ip=self.ip, port=self.port, auto_scan=True, auto_connect=True
            )
        except Exception as exc:
            raise HardwareCommunicationError(f"初始化机械臂驱动失败: {exc}") from exc

        if not self._driver.is_connected:
            raise HardwareCommunicationError(
                f"无法连接机械臂 {self.ip}:{self.port}"
            )
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
        self._connected = False

    def _ensure_ready(self) -> None:
        if self._stopped:
            raise SafetyViolationError("机械臂处于急停/锁定状态，请先复位")
        if not self._connected:
            raise HardwareCommunicationError("机械臂尚未连接，请先调用 connect()")

    def get_angles(self) -> List[float]:
        self._ensure_ready()
        if self.simulation:
            return list(self._sim_angles)
        try:
            vals = self._driver.get_angles(allow_cache=False)
        except TypeError:
            vals = self._driver.get_angles()
        if not is_valid_vector(vals, 6):
            raise HardwareCommunicationError(f"读取关节角度失败: {vals!r}")
        return [float(v) for v in vals[:6]]

    def get_coords(self) -> List[float]:
        self._ensure_ready()
        if self.simulation:
            return list(self._sim_coords)
        try:
            vals = self._driver.get_coords(allow_cache=False)
        except TypeError:
            vals = self._driver.get_coords()
        if not is_valid_vector(vals, 6):
            raise HardwareCommunicationError(f"读取笛卡尔坐标失败: {vals!r}")
        return [float(v) for v in vals[:6]]

    def get_gripper_status(self) -> Dict[str, Any]:
        self._ensure_ready()
        if self.simulation:
            angle = self._sim_gripper
            return {
                "angle": angle,
                "status_code": 0,
                "status_desc": "Simulation" if angle > 10 else "SimulationClosed",
            }
        fb = self._driver.get_gripper_feedback()
        val = fb.get("value")
        status = fb.get("status", -1)
        desc = "Moving/Active" if status == 1 else "Reached" if status == 0 else "Unknown"
        return {
            "angle": val,
            "status_code": status,
            "status_desc": desc,
        }

    def move_angles(self, angles: List[float], speed: int = 20) -> bool:
        self._ensure_ready()
        try:
            spd = validate_speed(speed)
            vec = validate_vector(angles, expected_len=6, name="angles")
            for idx, (val, lo, hi) in enumerate(
                zip(vec, PRO450_LIMITS["angles_min"], PRO450_LIMITS["angles_max"])
            ):
                if val < lo or val > hi:
                    raise ValueError(f"J{idx+1}={val} 超出安全限位 [{lo}, {hi}]")
            clean = clamp_angles(vec)
        except ValueError as exc:
            raise SafetyViolationError(str(exc)) from exc

        if self.require_human_confirmation:
            # 预留人工确认钩子；测试与工具默认关闭
            pass

        if self.simulation:
            self._sim_angles = list(clean)
            return True

        ok = self._driver.send_angles(clean, spd)
        if not ok:
            raise HardwareCommunicationError(f"send_angles 未确认到位: {clean}")
        return True

    def gripper_open(self, speed: int = 50) -> bool:
        self._ensure_ready()
        if self.simulation:
            self._sim_gripper = self.GRIPPER_OPEN_PERCENT
            return True
        ok = self._driver.set_gripper_value(self.GRIPPER_OPEN_PERCENT, speed=int(speed))
        if not ok:
            raise HardwareCommunicationError("夹爪张开指令执行失败")
        return True

    def gripper_close(self, speed: int = 50) -> bool:
        self._ensure_ready()
        if self.simulation:
            self._sim_gripper = self.GRIPPER_CLOSE_PERCENT
            return True
        ok = self._driver.set_gripper_value(self.GRIPPER_CLOSE_PERCENT, speed=int(speed))
        if not ok:
            raise HardwareCommunicationError("夹爪闭合指令执行失败")
        return True

    def stop(self) -> bool:
        self._stopped = True
        if self.simulation or self._driver is None:
            return True
        return bool(self._driver.emergency_stop())

    def clear_stop(self) -> None:
        self._stopped = False
        if self._driver is not None and hasattr(self._driver, "clear_stop"):
            self._driver.clear_stop()
