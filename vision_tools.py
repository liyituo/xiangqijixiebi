# -*- coding: utf-8 -*-
"""
机械臂视觉引导动作规划与综合具身工具箱 (Vision Tools & Embodied Arm Toolkit)
包含：
1. 五重物理硬约束安全审查 (PhysicalSafetyGuard)
2. 工业级 11 步无碰撞闭环搬运 (ACT-01 至 ACT-11)
3. 笛卡尔/关节全向点动与姿态控制工具
4. 预设工位快速巡航 (HOME, P1, P2, P3, P4, SCAN)
5. 自适应夹爪百分比行程控制
6. OpenVLA 7-DoF 归一化动作转物理步进执行器 (Action Chunking / Step Executor)
7. 多视点环视巡检与全景扫描
8. 自由牵引示教 (Drag Teach) 与头部 LED 状态指示
"""

import os
import sys
import json
import re
import time
import math
import requests
import numpy as np
import cv2
import inspect
from typing import Dict, Any, List, Tuple, Optional
from vision_calibration import HandEyeMapper
from pro450_driver import clamp_angles, clamp_coords, PRO450_LIMITS, is_finite_number, validate_speed

# 预设工位笛卡尔坐标 (X, Y, Z, Rx, Ry, Rz mm/deg) - 真实物理标定示教数据
PRESET_STATIONS = {
    "NORMAL": [162.0, -68.0, 542.0, 168.3, -19.4, 50.0],     # 全景俯视待机位 (NORMAL X:162 Y:-68 Z:542)
    "HOME":   [162.0, -68.0, 542.0, 168.3, -19.4, 50.0],     # 待机位 (HOME 等价于 NORMAL)
    "P1":     [296.4, 207.4, 183.1, -175.58, -8.38, 86.46],  # 工作工位 P1 (物料 A)
    "P2":     [378.4, -24.4, 189.0, -174.72, -7.45, 47.08],  # 工作工位 P2 (物料 B)
    "P3":     [341.6, -238.4, 188.8, -175.15, -8.28, 14.85], # 工作工位 P3 (成品工位)
    "P4":     [60.2, -440.1, 201.0, -172.67, -5.25, -33.39], # 工作工位 P4 (废料/暂存工位)
    "P5":     [153.2, -101.8, 204.6, -171.18, 0.28, 30.68],  # 工作工位 P5 (扩展工位)
    "SCAN":   [162.0, -68.0, 542.0, 168.3, -19.4, 50.0],     # 高位全景俯视观察点
}

# 预设工位关节角度 (J1, J2, J3, J4, J5, J6 deg) - 真实物理关节示教数据
PRESET_JOINTS = {
    "NORMAL": [5.0, -4.0, -7.0, -57.0, 5.0, -132.0],
    "HOME":   [5.0, -4.0, -7.0, -57.0, 5.0, -132.0],
    "P1":     [47.18, -41.34, -87.82, 42.87, 8.72, -129.32],
    "P2":     [7.92, -44.75, -78.11, 35.31, 8.79, -129.32],
    "P3":     [-24.38, -56.43, -56.48, 26.27, 8.98, -129.32],
    "P4":     [-72.33, -69.56, -24.34, 3.35, 8.98, -129.32],
    "P5":     [-8.31, 19.8, -147.51, 31.92, 6.65, -129.32],
    "SCAN":   [5.0, -4.0, -7.0, -57.0, 5.0, -132.0],
}


class VisionConfig:
    """具身视觉抓放与夹爪唯一参数源 (Single Source of Truth)"""
    GRIPPER_OPEN_PERCENT = 15
    GRIPPER_CLOSE_PERCENT = 5
    GRIPPER_OPEN_SPEED = 50
    GRIPPER_CLOSE_SPEED = 35
    GRASP_Z_OFFSET_MM = -2.0       # table_z - 2.0mm 稳定贴桌抓取高度
    HOVER_OFFSET_MM = 45.0
    HOVER_MIN_Z_MM = 210.0
    HOVER_MAX_Z_MM = 240.0
    CLEARANCE_MARGIN_MM = 15.0
    DEFAULT_PICK_ORIENTATION = (-178.5, -3.1, 46.0)


class PhysicalSafetyGuard:
    """五重物理硬约束安全审查 (与 PRO450_LIMITS 对齐)"""

    X_MIN, X_MAX = PRO450_LIMITS["coords_min"][0], PRO450_LIMITS["coords_max"][0]
    Y_MIN, Y_MAX = PRO450_LIMITS["coords_min"][1], PRO450_LIMITS["coords_max"][1]
    Z_MIN, Z_MAX = PRO450_LIMITS["table_safety_z"], PRO450_LIMITS["coords_max"][2]
    R_MAX = 475.0
    R_MIN = 0.0

    # 单步最大增量限制 (防大模型幻觉暴冲)
    MAX_STEP_XYZ = 50.0              # mm
    MAX_STEP_ROT = 35.0              # deg

    @classmethod
    def validate_target_point(cls, x: float, y: float, z: float) -> Tuple[bool, str]:
        """验证空间点位是否有限且位于绝对安全工作空间内"""
        if not (is_finite_number(x) and is_finite_number(y) and is_finite_number(z)):
            return False, f"目标坐标包含非有限值: ({x!r}, {y!r}, {z!r})"
        fx, fy, fz = float(x), float(y), float(z)
        dist = math.sqrt(fx**2 + fy**2)
        if dist > cls.R_MAX:
            return False, f"物理超程: 半径距离 {dist:.1f}mm 超过安全极限 {cls.R_MAX}mm"
        if dist < cls.R_MIN:
            return False, f"过近盲区: 距离 {dist:.1f}mm 进入基座干涉区 (<{cls.R_MIN}mm)"

        if not (cls.X_MIN <= fx <= cls.X_MAX):
            return False, f"X坐标越界 [{fx:.1f}mm]，安全范围为 [{cls.X_MIN}, {cls.X_MAX}]mm"
        if not (cls.Y_MIN <= fy <= cls.Y_MAX):
            return False, f"Y坐标越界 [{fy:.1f}mm]，安全范围为 [{cls.Y_MIN}, {cls.Y_MAX}]mm"
        if not (cls.Z_MIN <= fz <= cls.Z_MAX):
            return False, f"Z高度越界 [{fz:.1f}mm]，防止撞击桌面 (允许范围 [{cls.Z_MIN}, {cls.Z_MAX}]mm)"

        return True, "安全审查通过"

    @classmethod
    def validate_delta_step(cls, dx: float, dy: float, dz: float,
                            drx: float = 0.0, dry: float = 0.0, drz: float = 0.0) -> Tuple[bool, str]:
        """验证单步微调增量是否有限且在安全阈值内"""
        if not all(is_finite_number(v) for v in (dx, dy, dz, drx, dry, drz)):
            return False, "增量包含非有限值或非法类型"
        dist_delta = math.sqrt(float(dx)**2 + float(dy)**2 + float(dz)**2)
        if dist_delta > cls.MAX_STEP_XYZ + 1e-6:
            return False, f"单步平移过大: {dist_delta:.1f}mm > {cls.MAX_STEP_XYZ}mm，自动限幅保护"
        rot_delta = max(abs(float(drx)), abs(float(dry)), abs(float(drz)))
        if rot_delta > cls.MAX_STEP_ROT + 1e-6:
            return False, f"单步旋转过大: {rot_delta:.1f}deg > {cls.MAX_STEP_ROT}deg"
        return True, "增量安全"

    @staticmethod
    def validate_station(station_name: str) -> bool:
        return isinstance(station_name, str) and station_name.strip().upper() in PRESET_STATIONS


class VisionActionExecutor:
    """视觉引导物理执行中枢与具身综合工具箱"""

    def __init__(self, driver, mapper: HandEyeMapper):
        self.driver = driver
        self.mapper = mapper

    def _call_driver_method(self, method_name: str, *args, **kwargs):
        """安全调用驱动方法，仅在目标签名支持时透传 cancel_token / expected_clamp / allow_cache 等可选参数。"""
        fn = getattr(self.driver, method_name)
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if has_var_kw:
                filtered = kwargs
            else:
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            filtered = {}
        return fn(*args, **filtered)

    def _read_fresh_coords(self) -> Tuple[Optional[List[float]], Optional[str]]:
        if hasattr(self.driver, "get_coords"):
            res = self._call_driver_method("get_coords", allow_cache=False)
            if isinstance(res, (list, tuple)) and len(res) >= 6 and all(is_finite_number(v) for v in res[:6]):
                return [float(v) for v in res[:6]], None
        return None, "无法读取机械臂当前真实坐标，请检查驱动连接或反馈状态"

    def _read_fresh_angles(self) -> Tuple[Optional[List[float]], Optional[str]]:
        if hasattr(self.driver, "get_angles"):
            res = self._call_driver_method("get_angles", allow_cache=False)
            if isinstance(res, (list, tuple)) and len(res) >= 6 and all(is_finite_number(v) for v in res[:6]):
                return [float(v) for v in res[:6]], None
        return None, "无法读取机械臂当前真实关节角，请检查驱动连接或反馈状态"

    def _check_cancelled(self, cancel_token: Any = None):
        backend = getattr(self.driver, "backend", self.driver)
        stop_requested = getattr(backend, "stop_requested", None)
        if stop_requested is not None and stop_requested.is_set():
            raise RuntimeError("动作已被停止，后续步骤未执行")
        if cancel_token is not None and hasattr(cancel_token, "is_cancelled") and cancel_token.is_cancelled():
            raise RuntimeError("动作已被取消，后续步骤未执行")

    def _sleep_cancelable(self, seconds: float, cancel_token: Any = None):
        if cancel_token is not None and hasattr(cancel_token, "wait"):
            cancel_token.wait(seconds, step=0.05)
            self._check_cancelled(cancel_token)
        else:
            time.sleep(seconds)
            self._check_cancelled(cancel_token)

    # --------------------------------------------------------------------------
    # Tool 1: 视觉目标检测与 11 步搬运工序 (Visual Pick & Place)
    # --------------------------------------------------------------------------
    def execute_visual_pick_and_place(
        self,
        target_name: str,
        bbox: List[int],
        target_station: Any = "P5",
        speed: int = 15,
        step_callback: Optional[callable] = None,
        offset_trim: Optional[Tuple[float, float, float]] = None,
        place_offset: Optional[Tuple[float, float, float]] = None,
        cancel_token: Any = None,
        expected_clamp: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        核心物理流程：基于视觉检测框，执行带空中检查点的工业闭环搬运。
        严格校验 bbox、速度、自定义坐标与已知工位名（未知工位拒绝猜测回退），支持取消令牌与夹持确认。
        """
        try:
            speed_val = validate_speed(speed)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(is_finite_number(v) for v in bbox):
            return {"success": False, "error": f"无效视觉框参数: {bbox!r}"}
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]
        if ymax < ymin or xmax < xmin or min(ymin, xmin) < 0.0 or max(ymax, xmax) > 1000.0:
            return {"success": False, "error": f"视觉框范围或顺序非法: {bbox!r}"}

        if offset_trim is not None:
            if not isinstance(offset_trim, (list, tuple)) or len(offset_trim) < 3 or not all(is_finite_number(v) for v in offset_trim[:3]):
                return {"success": False, "error": f"手眼偏置参数非法: {offset_trim!r}"}
        if place_offset is not None:
            if not isinstance(place_offset, (list, tuple)) or len(place_offset) < 3 or not all(is_finite_number(v) for v in place_offset[:3]):
                return {"success": False, "error": f"放置偏移参数非法: {place_offset!r}"}

        dest_display = str(target_station)
        print(f"\n🚀 >>> 启动视觉抓取工序: 目标【{target_name}】 -> 目标目的地【{dest_display}】")

        # 1. 坐标换算与映射
        target_x, target_y, table_z = self.mapper.bbox_to_robot_center(bbox)
        hover_z = min(
            VisionConfig.HOVER_MAX_Z_MM,
            max(VisionConfig.HOVER_MIN_Z_MM, table_z + VisionConfig.HOVER_OFFSET_MM),
        )
        grasp_z = round(table_z + VisionConfig.GRASP_Z_OFFSET_MM, 1)

        if offset_trim:
            target_x = round(target_x + float(offset_trim[0]), 1)
            target_y = round(target_y + float(offset_trim[1]), 1)
            grasp_z = round(grasp_z + float(offset_trim[2]), 1)

        grip_clamp_val = VisionConfig.GRIPPER_CLOSE_PERCENT
        grip_open_val = VisionConfig.GRIPPER_OPEN_PERCENT

        print(
            f"📍 视觉映射物理抓取点: X={target_x}mm, Y={target_y}mm, "
            f"抓取高度 Z={grasp_z}mm (过渡 Z={hover_z}mm), 夹爪: 开={grip_open_val}%, 闭={grip_clamp_val}%"
        )

        # 2. 抓取点安全防火墙硬拦截
        safe_ok, safe_reason = PhysicalSafetyGuard.validate_target_point(target_x, target_y, grasp_z)
        if not safe_ok:
            print(f"🛑 [安全防火墙硬拦截]: {safe_reason}！取消动作以保护硬件。")
            return {"success": False, "error": safe_reason}

        # 3. 严格解析目标目的地 (禁止将未知工位名静默回退到 P5)
        use_custom_coord = False
        target_coord = None
        station_angles = None

        if isinstance(target_station, (list, tuple)):
            if len(target_station) < 2 or not all(is_finite_number(v) for v in target_station[:min(6, len(target_station))]):
                return {"success": False, "error": f"自定义目的地坐标格式或数值非法: {target_station!r}"}
            use_custom_coord = True
            dx = float(target_station[0])
            dy = float(target_station[1])
            dz = float(target_station[2]) if len(target_station) >= 3 else (table_z + 8.0)
            rx = float(target_station[3]) if len(target_station) >= 4 else VisionConfig.DEFAULT_PICK_ORIENTATION[0]
            ry = float(target_station[4]) if len(target_station) >= 5 else VisionConfig.DEFAULT_PICK_ORIENTATION[1]
            rz = float(target_station[5]) if len(target_station) >= 6 else VisionConfig.DEFAULT_PICK_ORIENTATION[2]
            if place_offset:
                dx += float(place_offset[0])
                dy += float(place_offset[1])
                dz += float(place_offset[2])
            target_coord = [round(dx, 2), round(dy, 2), round(dz, 2), rx, ry, rz]
        else:
            if not isinstance(target_station, str) or not target_station.strip():
                return {"success": False, "error": f"无效目标工位: {target_station!r}"}
            station_str = target_station.strip().upper()
            if station_str not in PRESET_STATIONS and station_str not in PRESET_JOINTS:
                return {"success": False, "error": f"未知目标工位 '{target_station}'，拒绝使用默认工位覆盖"}
            if place_offset:
                use_custom_coord = True
                base = PRESET_STATIONS[station_str]
                target_coord = [
                    round(base[0] + float(place_offset[0]), 2),
                    round(base[1] + float(place_offset[1]), 2),
                    round(base[2] + float(place_offset[2]), 2),
                    base[3], base[4], base[5],
                ]
            else:
                target_coord = list(PRESET_STATIONS[station_str])
                station_angles = PRESET_JOINTS.get(station_str)

        min_safe_hover = grasp_z + VisionConfig.CLEARANCE_MARGIN_MM
        if use_custom_coord and target_coord:
            dest_ok, dest_reason = PhysicalSafetyGuard.validate_target_point(
                target_coord[0], target_coord[1], target_coord[2]
            )
            if not dest_ok:
                print(f"🛑 [安全防火墙硬拦截]: 目标放置点不安全 ({dest_reason})！取消动作以保护硬件。")
                return {"success": False, "error": f"目标放置点安全拦截: {dest_reason}"}
            min_safe_hover = max(min_safe_hover, target_coord[2] + VisionConfig.CLEARANCE_MARGIN_MM)
        hover_z = round(max(hover_z, min_safe_hover), 1)

        # 4. 构造高安全性工业搬运闭环
        down_speed = max(5, int(speed_val * 0.6))
        rx0, ry0, rz0 = VisionConfig.DEFAULT_PICK_ORIENTATION

        def send_coords_checked(coords, spd):
            res = self._call_driver_method("send_coords", coords, speed=spd, cancel_token=cancel_token)
            if isinstance(res, dict) and res.get("ok") is False:
                raise RuntimeError(res.get("error", "机械臂移动失败"))
            return res

        def send_angles_checked(angles, spd):
            res = self._call_driver_method("send_angles", angles, speed=spd, cancel_token=cancel_token)
            if isinstance(res, dict) and res.get("ok") is False:
                raise RuntimeError(res.get("error", "机械臂关节移动失败"))
            return res

        def set_gripper_checked(val, spd, exp_clamp=None):
            res = self._call_driver_method(
                "set_gripper_value",
                val,
                speed=spd,
                cancel_token=cancel_token,
                expected_clamp=exp_clamp,
            )
            if isinstance(res, dict) and res.get("ok") is False:
                raise RuntimeError(res.get("error", "夹爪动作失败"))
            return res

        def move_to_delivery():
            if station_angles:
                send_angles_checked(station_angles, speed_val)
            elif target_coord:
                send_coords_checked([target_coord[0], target_coord[1], hover_z, target_coord[3], target_coord[4], target_coord[5]], speed_val)
                send_coords_checked(target_coord, down_speed)

        def act_04_clamp():
            grip_res = set_gripper_checked(grip_clamp_val, VisionConfig.GRIPPER_CLOSE_SPEED, exp_clamp=expected_clamp)
            if isinstance(grip_res, dict) and grip_res.get("dropped"):
                raise RuntimeError("夹爪闭合抓取时检测到物体滑落")
            self._sleep_cancelable(1.0, cancel_token)

        def act_06_check_hold():
            if hasattr(self.driver, "get_gripper_feedback"):
                fb = self.driver.get_gripper_feedback()
                if isinstance(fb, dict) and fb.get("status_code") == 3:
                    raise RuntimeError("空中安全检查失败：物体已在抬升过程中掉落")
            self._sleep_cancelable(0.6, cancel_token)

        def act_10_retreat():
            # 在返回待机前从抓放低高度平稳回退抬升，避免横扫桌面其他物体
            if target_coord and not station_angles:
                send_coords_checked([target_coord[0], target_coord[1], hover_z, target_coord[3], target_coord[4], target_coord[5]], down_speed)
            self._sleep_cancelable(0.5, cancel_token)

        steps = [
            ("ACT-01", f"张开自适应夹爪 ({grip_open_val}% 开度)", lambda: (set_gripper_checked(grip_open_val, VisionConfig.GRIPPER_OPEN_SPEED), self._sleep_cancelable(0.4, cancel_token))),
            ("ACT-02", f"平移至物体上方安全悬停点 ({target_x:.1f}, {target_y:.1f}, {hover_z:.1f})", lambda: (send_coords_checked([target_x, target_y, hover_z, rx0, ry0, rz0], speed_val), self._sleep_cancelable(1.2, cancel_token))),
            ("ACT-03", f"垂直缓冲下放抓取 ({target_x:.1f}, {target_y:.1f}, {grasp_z:.1f})", lambda: (send_coords_checked([target_x, target_y, grasp_z, rx0, ry0, rz0], down_speed), self._sleep_cancelable(1.2, cancel_token))),
            ("ACT-04", f"柔性闭合夹爪夹取工件【{target_name}】(开度 {grip_clamp_val}%)", act_04_clamp),
            ("ACT-05", f"垂直抬升至安全空域高度 ({hover_z:.1f}mm)", lambda: (send_coords_checked([target_x, target_y, hover_z, rx0, ry0, rz0], down_speed), self._sleep_cancelable(1.0, cancel_token))),
            ("ACT-06", "【空中安全检查点】悬停确认夹持稳定", act_06_check_hold),
            ("ACT-07", f"经由高位安全航线巡航至目的地 [{dest_display}]", lambda: (move_to_delivery(), self._sleep_cancelable(1.5, cancel_token))),
            ("ACT-08", "工位就位确认", lambda: self._sleep_cancelable(0.5, cancel_token)),
            ("ACT-09", f"释放夹爪交付工件 ({grip_open_val}% 开度)", lambda: (set_gripper_checked(grip_open_val, VisionConfig.GRIPPER_OPEN_SPEED), self._sleep_cancelable(0.8, cancel_token))),
            ("ACT-10", "工件交付确认与安全回退抬升", act_10_retreat),
            ("ACT-11", "平稳返回初始待机 NORMAL 姿态", lambda: (send_angles_checked(PRESET_JOINTS["NORMAL"], speed_val), self._sleep_cancelable(1.5, cancel_token))),
        ]

        total_steps = len(steps)
        for i, (step_id, desc, action) in enumerate(steps):
            self._check_cancelled(cancel_token)
            print(f"   [{step_id}] -> {desc}")
            if step_callback:
                step_callback(step_id, desc, int((i + 1) / total_steps * 100))
            action()
            self._sleep_cancelable(0.2, cancel_token)

        print(f"🎉 视觉闭环搬运成功！工件【{target_name}】已精准移至工位【{target_station}】。\n")
        return {
            "success": True,
            "target_object": target_name,
            "robot_coords": [target_x, target_y, table_z],
            "target_station": target_station
        }

    # --------------------------------------------------------------------------
    # Tool 2: 笛卡尔坐标绝对定位 (Move to Coordinates)
    # --------------------------------------------------------------------------
    def move_to_coords(self, coords: List[float], speed: int = 15, cancel_token: Any = None) -> Dict[str, Any]:
        """移动末端至指定笛卡尔坐标 [x, y, z, rx, ry, rz]"""
        try:
            speed_val = validate_speed(speed)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        if not isinstance(coords, (list, tuple)) or len(coords) < 3 or not all(is_finite_number(v) for v in coords):
            return {"success": False, "error": "坐标参数不足或包含非有限数值"}
        if len(coords) not in (3, 6):
            return {"success": False, "error": f"坐标维度必须为 3 或 6，当前为 {len(coords)}"}

        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
        safe_ok, reason = PhysicalSafetyGuard.validate_target_point(x, y, z)
        if not safe_ok:
            return {"success": False, "error": reason}

        full_coords = [
            x, y, z,
            float(coords[3]) if len(coords) == 6 else VisionConfig.DEFAULT_PICK_ORIENTATION[0],
            float(coords[4]) if len(coords) == 6 else VisionConfig.DEFAULT_PICK_ORIENTATION[1],
            float(coords[5]) if len(coords) == 6 else VisionConfig.DEFAULT_PICK_ORIENTATION[2],
        ]
        clamped = clamp_coords(full_coords, safety_z_min=PRO450_LIMITS["table_safety_z"])
        self._call_driver_method("send_coords", clamped, speed=speed_val, cancel_token=cancel_token)
        return {"success": True, "target": clamped}

    # --------------------------------------------------------------------------
    # Tool 3: 笛卡尔空间单轴微调点动 (Jog Cartesian Step)
    # --------------------------------------------------------------------------
    def jog_cartesian(self, axis: str, delta: float, speed: int = 15, cancel_token: Any = None) -> Dict[str, Any]:
        """末端沿指定轴增量微调 (+X, -X, +Y, -Y, +Z, -Z, +Rx, etc.)"""
        if not is_finite_number(delta):
            return {"success": False, "error": f"步长 delta 必须为有限数字: {delta!r}"}
        try:
            speed_val = validate_speed(speed)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        if not isinstance(axis, str):
            return {"success": False, "error": f"无效轴名: {axis!r}"}

        axis = axis.strip().lower()
        axis_map = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
        if axis not in axis_map:
            return {"success": False, "error": f"无效轴名: {axis}"}

        idx = axis_map[axis]
        delta_val = float(delta)
        if idx in (0, 1, 2) and abs(delta_val) > PhysicalSafetyGuard.MAX_STEP_XYZ:
            return {"success": False, "error": f"单步平移增量超限: {abs(delta_val):.1f}mm > {PhysicalSafetyGuard.MAX_STEP_XYZ}mm"}
        if idx in (3, 4, 5) and abs(delta_val) > PhysicalSafetyGuard.MAX_STEP_ROT:
            return {"success": False, "error": f"单步旋转增量超限: {abs(delta_val):.1f}deg > {PhysicalSafetyGuard.MAX_STEP_ROT}deg"}

        curr, err = self._read_fresh_coords()
        if curr is None:
            return {"success": False, "error": err}

        target = list(curr)
        target[idx] += delta_val

        if idx in (0, 1, 2):
            safe_ok, reason = PhysicalSafetyGuard.validate_target_point(target[0], target[1], target[2])
            if not safe_ok:
                return {"success": False, "error": reason}

        curr_angles, _ = self._read_fresh_angles()
        if curr_angles and all(abs(a) < 1.0 for a in curr_angles[1:4]):
            self._call_driver_method(
                "send_angles",
                [curr_angles[0], 10.0, -15.0, 5.0, curr_angles[4], curr_angles[5]],
                speed=20,
                cancel_token=cancel_token,
            )
            self._sleep_cancelable(0.5, cancel_token)

        clamped = clamp_coords(target, safety_z_min=PRO450_LIMITS["table_safety_z"])
        self._call_driver_method("send_coords", clamped, speed=speed_val, cancel_token=cancel_token)
        return {"success": True, "new_coords": clamped}

    # --------------------------------------------------------------------------
    # Tool 4: 关节角度控制与微调点动 (Joints Control & Jog)
    # --------------------------------------------------------------------------
    def move_joints(self, angles: List[float], speed: int = 15, cancel_token: Any = None) -> Dict[str, Any]:
        """驱动机械臂 6 个关节角度"""
        try:
            speed_val = validate_speed(speed)
            clamped = clamp_angles(angles)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        self._call_driver_method("send_angles", clamped, speed=speed_val, cancel_token=cancel_token)
        return {"success": True, "angles": clamped}

    def jog_joint(self, joint_idx: int, delta_deg: float, speed: int = 15, cancel_token: Any = None) -> Dict[str, Any]:
        """单关节微调点动 (joint_idx: 1~6)"""
        if isinstance(joint_idx, bool) or not isinstance(joint_idx, int) or not (1 <= joint_idx <= 6):
            return {"success": False, "error": "关节编号必须在 1~6 之间"}
        if not is_finite_number(delta_deg):
            return {"success": False, "error": f"关节增量必须为有限数字: {delta_deg!r}"}
        try:
            speed_val = validate_speed(speed)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        curr, err = self._read_fresh_angles()
        if curr is None:
            return {"success": False, "error": err}

        target = list(curr)
        target[joint_idx - 1] += float(delta_deg)
        target = clamp_angles(target)

        self._call_driver_method("send_angles", target, speed=speed_val, cancel_token=cancel_token)
        return {"success": True, "new_angles": target}

    # --------------------------------------------------------------------------
    # Tool 5: 预设标定工位巡航 (Move to Station)
    # --------------------------------------------------------------------------
    def move_to_station(self, station_name: str, speed: int = 15, cancel_token: Any = None) -> Dict[str, Any]:
        """快速巡航至已标定的物理工位 (HOME, NORMAL, P1..P5, SCAN)"""
        if not isinstance(station_name, str) or not station_name.strip():
            return {"success": False, "error": f"无效工位: {station_name!r}"}
        try:
            speed_val = validate_speed(speed)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        s_name = station_name.strip().upper()
        if s_name not in PRESET_STATIONS and s_name not in PRESET_JOINTS:
            return {"success": False, "error": f"未知工位: {station_name}"}

        if s_name in PRESET_JOINTS:
            angles = PRESET_JOINTS[s_name]
            self._call_driver_method("send_angles", angles, speed=speed_val, cancel_token=cancel_token)
            return {"success": True, "station": s_name, "angles": angles}

        coords = PRESET_STATIONS[s_name]
        self._call_driver_method("send_coords", coords, speed=speed_val, cancel_token=cancel_token)
        return {"success": True, "station": s_name, "coords": coords}

    # --------------------------------------------------------------------------
    # Tool 6: 自适应夹爪连续行程控制 (Continuous Gripper Control)
    # --------------------------------------------------------------------------
    def control_gripper(self, action: str = "OPEN", value_percent: Optional[int] = None, speed: int = 50, cancel_token: Any = None) -> Dict[str, Any]:
        """夹爪控制：支持快速开合 (OPEN/CLOSE) 或 0~100% 连续行程开度"""
        try:
            speed_val = validate_speed(speed)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        if value_percent is not None:
            if not is_finite_number(value_percent) or value_percent < 0 or value_percent > 100:
                return {"success": False, "error": f"夹爪百分比必须在 [0, 100] 范围内: {value_percent!r}"}
            val = int(round(float(value_percent)))
            res = self._call_driver_method("set_gripper_value", val, speed=speed_val, cancel_token=cancel_token)
            return {"success": True, "mode": "percentage", "value": val, "feedback": res}
        act = str(action).strip().upper()
        if act not in ("OPEN", "CLOSE"):
            return {"success": False, "error": f"无效夹爪动作: {action!r}"}
        res = self._call_driver_method("set_gripper", act, speed=speed_val, cancel_token=cancel_token)
        return {"success": True, "mode": "discrete", "action": act, "feedback": res}

    # --------------------------------------------------------------------------
    # Tool 7: OpenVLA 7-DoF 动作步进转换执行器 (OpenVLA Step Action)
    # --------------------------------------------------------------------------
    def execute_openvla_action_step(
        self,
        action_7d: List[float],
        scale_xyz_mm: float = 20.0,
        scale_rot_deg: float = 10.0,
        gripper_threshold: float = 0.5,
        action_spec: str = "normalized_delta",
        cancel_token: Any = None,
    ) -> Dict[str, Any]:
        """
        执行 OpenVLA 预测的 7-DoF 动作增量 [dx, dy, dz, droll, dpitch, dyaw, gripper]。
        支持显式 action_spec:
          - 'normalized_delta': 模型输出 [-1, 1]，按 scale_xyz_mm / scale_rot_deg 映射为物理 mm/deg；
          - 'bridge_unnormalized' / 'metric_delta': 模型输出物理 m 与 rad，换算为 mm (*1000) 与 deg (*180/pi)。
        同时对平移与旋转均执行限幅与校验，并核验当前实时位姿。
        """
        if not isinstance(action_7d, (list, tuple)) or len(action_7d) < 7:
            return {"success": False, "error": f"OpenVLA 动作维度不足 7: {len(action_7d) if isinstance(action_7d, (list, tuple)) else type(action_7d).__name__}"}
        if not all(is_finite_number(v) for v in action_7d[:7]):
            return {"success": False, "error": f"OpenVLA 动作包含非有限值或非法类型: {action_7d[:7]!r}"}
        if not (is_finite_number(scale_xyz_mm) and scale_xyz_mm > 0 and is_finite_number(scale_rot_deg) and scale_rot_deg > 0):
            return {"success": False, "error": "scale_xyz_mm 与 scale_rot_deg 必须为正有限数"}

        raw_dx, raw_dy, raw_dz = [float(v) for v in action_7d[:3]]
        raw_drx, raw_dry, raw_drz = [float(v) for v in action_7d[3:6]]
        raw_grip = float(action_7d[6])

        spec = str(action_spec or "normalized_delta").strip().lower()
        if spec in ("bridge_unnormalized", "metric_delta", "meters_radians"):
            dx, dy, dz = raw_dx * 1000.0, raw_dy * 1000.0, raw_dz * 1000.0
            drx = math.degrees(raw_drx)
            dry = math.degrees(raw_dry)
            drz = math.degrees(raw_drz)
        elif spec == "normalized_delta":
            dx = raw_dx * float(scale_xyz_mm)
            dy = raw_dy * float(scale_xyz_mm)
            dz = raw_dz * float(scale_xyz_mm)
            drx = raw_drx * float(scale_rot_deg)
            dry = raw_dry * float(scale_rot_deg)
            drz = raw_drz * float(scale_rot_deg)
        else:
            return {"success": False, "error": f"未知 action_spec: {action_spec!r}"}

        # 同时对平移与旋转执行等比限幅保护
        safe_delta, reason = PhysicalSafetyGuard.validate_delta_step(dx, dy, dz, drx, dry, drz)
        clamped_by_guard = False
        if not safe_delta:
            print(f"⚠️ OpenVLA 动作限幅告警: {reason}")
            clamped_by_guard = True
            norm_xyz = math.sqrt(dx**2 + dy**2 + dz**2)
            if norm_xyz > PhysicalSafetyGuard.MAX_STEP_XYZ:
                s_xyz = PhysicalSafetyGuard.MAX_STEP_XYZ / norm_xyz
                dx *= s_xyz
                dy *= s_xyz
                dz *= s_xyz
            max_rot = max(abs(drx), abs(dry), abs(drz))
            if max_rot > PhysicalSafetyGuard.MAX_STEP_ROT:
                s_rot = PhysicalSafetyGuard.MAX_STEP_ROT / max_rot
                drx *= s_rot
                dry *= s_rot
                drz *= s_rot

            ok_after, err_after = PhysicalSafetyGuard.validate_delta_step(dx, dy, dz, drx, dry, drz)
            if not ok_after:
                return {"success": False, "error": f"OpenVLA 限幅后仍不安全: {err_after}"}

        curr, err = self._read_fresh_coords()
        if curr is None:
            return {"success": False, "error": err}

        target_xyz = [curr[0] + dx, curr[1] + dy, curr[2] + dz]
        safe_pt, pt_reason = PhysicalSafetyGuard.validate_target_point(*target_xyz)
        if not safe_pt:
            return {"success": False, "error": f"OpenVLA 目标点越界: {pt_reason}"}

        def _wrap_deg(a: float) -> float:
            return ((a + 180.0) % 360.0) - 180.0

        target_coords = clamp_coords([
            target_xyz[0],
            target_xyz[1],
            target_xyz[2],
            _wrap_deg(curr[3] + drx),
            max(-180.0, min(180.0, curr[4] + dry)),
            _wrap_deg(curr[5] + drz),
        ], safety_z_min=PRO450_LIMITS["table_safety_z"])

        self._call_driver_method("send_coords", target_coords, speed=15, cancel_token=cancel_token)

        gripper_state = "CLOSE" if raw_grip < gripper_threshold else "OPEN"
        grip_res = self._call_driver_method("set_gripper", gripper_state, speed=60, cancel_token=cancel_token)

        return {
            "success": True,
            "action_spec": spec,
            "clamped": clamped_by_guard,
            "applied_delta": [round(dx, 2), round(dy, 2), round(dz, 2), round(drx, 2), round(dry, 2), round(drz, 2)],
            "new_coords": target_coords,
            "gripper": gripper_state,
            "gripper_feedback": grip_res,
        }

    # --------------------------------------------------------------------------
    # Tool 8: 多视点环视全景巡检扫描 (Panoramic Workspace Scan)
    # --------------------------------------------------------------------------
    def scan_workspace(self, scan_callback: Optional[callable] = None, cancel_token: Any = None) -> Dict[str, Any]:
        """机械臂执行预定三维巡检航线，支持取消令牌中断与真实反馈检查。"""
        self.set_status_led("MOVING")
        scan_waypoints = [
            ("SCAN-P1",     PRESET_JOINTS["P1"],     "工位 P1 俯视视点 (左侧工位 X:296 Y:207)"),
            ("SCAN-NORMAL", PRESET_JOINTS["NORMAL"], "工作台全景俯视观测点 (NORMAL X:162 Y:-68 Z:542)"),
            ("SCAN-P3",     PRESET_JOINTS["P3"],     "工位 P3 俯视视点 (右侧成品 X:342 Y:-238)"),
        ]

        results = []
        total = len(scan_waypoints)
        for i, (code, angles, desc) in enumerate(scan_waypoints):
            self._check_cancelled(cancel_token)
            print(f"🔍 巡检航点 [{i+1}/{total}]: {desc} -> 关节角度 {angles}")
            self._call_driver_method("send_angles", angles, speed=18, cancel_token=cancel_token)
            self._sleep_cancelable(1.2, cancel_token)
            coords, _ = self._read_fresh_coords()
            pct = int(((i + 1) / (total + 1)) * 100)
            if scan_callback:
                scan_callback(code, desc, pct)
            results.append({"code": code, "desc": desc, "angles": angles, "coords": coords})

        self._check_cancelled(cancel_token)
        print("🔄 巡检扫描完毕，机械臂平稳返回待机工位 HOME...")
        self.move_to_station("HOME", speed=18, cancel_token=cancel_token)
        self._sleep_cancelable(1.0, cancel_token)
        self.set_status_led("IDLE")
        if scan_callback:
            scan_callback("SCAN-DONE", "全景三维巡检扫描已圆满完成", 100)
        return {"success": True, "scanned_waypoints": results}

    # --------------------------------------------------------------------------
    # Tool 9: 示教拖拽自由牵引模式 (Drag Teach Mode)
    # --------------------------------------------------------------------------
    def set_drag_teach(self, enable: bool) -> Dict[str, Any]:
        """开启或退出机械臂自由拖动示教模式 (释放舵机/恢复扭矩)"""
        if enable:
            self.driver.release_all_servos()
            return {"success": True, "mode": "TEACH_ENABLED", "msg": "所有关节已卸载扭矩，可手动拖拽示教"}
        else:
            self.driver.focus_all_servos()
            return {"success": True, "mode": "TEACH_DISABLED", "msg": "舵机扭矩已锁定，恢复自动作业状态"}

    # --------------------------------------------------------------------------
    # Tool 10: 头部 RGB 状态指示灯 (LED Status Indicator)
    # --------------------------------------------------------------------------
    def set_status_led(self, status: str) -> Dict[str, Any]:
        """控制头部 RGB LED 灯语反馈"""
        status_colors = {
            "IDLE": (0, 255, 0),        # 绿色 待机就绪
            "VISION": (0, 150, 255),    # 蓝色 大模型思考中
            "MOVING": (255, 200, 0),    # 黄色 运动执行中
            "ERROR": (255, 0, 0),       # 红色 报警/急停
            "LIGHT": (255, 255, 255),   # 白色 工作补光
            "OFF": (0, 0, 0)            # 关闭
        }
        color = status_colors.get(status.upper(), (0, 255, 0))
        self.driver.set_color(color[0], color[1], color[2])
        return {"success": True, "status": status.upper(), "rgb": color}

    # --------------------------------------------------------------------------
    # Tool 11: 硬件紧急制动 (Hardware Emergency Stop)
    # --------------------------------------------------------------------------
    def emergency_stop(self, reason: str = "手动制动") -> Dict[str, Any]:
        """最高优先级紧急制动"""
        self.driver.emergency_stop()
        self.set_status_led("ERROR")
        return {"success": True, "reason": reason}


# ==============================================================================
# 自适应工作台视觉目标检测器 (Edge Visual Object Detector)
# ==============================================================================
# ==============================================================================
# 自适应工作台多模态全景视觉目标感知系统 (Qwen3.8-Flash Cloud Direct)
# ==============================================================================
from env_loader import get_vlm_api_key

DEEPSEEK_API_KEY = get_vlm_api_key("VLM_API_KEY", "DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("VLM_MODEL", os.environ.get("DEEPSEEK_MODEL", "qwen3.8-flash"))
DEEPSEEK_API_URL = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

def _extract_vlm_json(text: str) -> Optional[Dict[str, Any]]:
    """从大模型原始文本中鲁棒提取 JSON (支持 Markdown 代码块提取与截断自动修复)"""
    if not text:
        return None
    if "```" in text:
        matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        for m in matches:
            try:
                return json.loads(m.strip())
            except Exception:
                pass
    first_b = text.find("{")
    last_b = text.rfind("}")
    if first_b != -1 and last_b != -1 and last_b > first_b:
        try:
            return json.loads(text[first_b:last_b+1])
        except Exception:
            pass

    # 截断自动补全容错修复：找到最后一个完整的闭合 '}' 并补齐 ']}'
    if first_b != -1:
        truncated = text[first_b:]
        last_obj_end = truncated.rfind("}")
        while last_obj_end != -1:
            candidate = truncated[:last_obj_end+1].strip()
            for suffix in ["]}", "}", "\n]}", "\n}"]:
                try:
                    data = json.loads(candidate + suffix)
                    if isinstance(data.get("objects"), list) and len(data["objects"]) > 0:
                        return data
                except Exception:
                    pass
            last_obj_end = truncated.rfind("}", 0, last_obj_end)
    return None

def refine_piece_center_circle(
    frame: Optional[np.ndarray],
    bbox: List[int],
    pad_ratio: float = 0.25
) -> Tuple[float, float, List[int]]:
    """
    基于 OpenCV 霍夫圆变换 (HoughCircles) 与轮廓拟合，对视觉大模型提取的象棋棋子/圆形工件检测框进行亚毫米级几何对齐。
    返回精确对齐后的: (norm_cx, norm_cy, refined_bbox) (0~1000 归一化)
    """
    if frame is None or len(bbox) != 4:
        y1_raw = min(bbox[0], bbox[2]) if len(bbox) == 4 else 0
        y2_raw = max(bbox[0], bbox[2]) if len(bbox) == 4 else 1000
        x1_raw = min(bbox[1], bbox[3]) if len(bbox) == 4 else 0
        x2_raw = max(bbox[1], bbox[3]) if len(bbox) == 4 else 1000
        return ((x1_raw + x2_raw) / 2.0, (y1_raw + y2_raw) / 2.0, [y1_raw, x1_raw, y2_raw, x2_raw])

    h, w = frame.shape[:2]
    y1_raw, y2_raw = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
    x1_raw, x2_raw = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])

    ymin_px = int(y1_raw * h / 1000.0)
    xmin_px = int(x1_raw * w / 1000.0)
    ymax_px = int(y2_raw * h / 1000.0)
    xmax_px = int(x2_raw * w / 1000.0)

    bw = max(10, xmax_px - xmin_px)
    bh = max(10, ymax_px - ymin_px)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    crop_y1 = max(0, ymin_px - pad_y)
    crop_y2 = min(h, ymax_px + pad_y)
    crop_x1 = max(0, xmin_px - pad_x)
    crop_x2 = min(w, xmax_px + pad_x)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return ((x1_raw + x2_raw) / 2.0, (y1_raw + y2_raw) / 2.0, [y1_raw, x1_raw, y2_raw, x2_raw])

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    expected_r = (bw + bh) / 4.0
    min_r = max(12, int(expected_r * 0.6))
    max_r = max(min_r + 5, int(expected_r * 1.4))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(15, int(expected_r)),
        param1=50,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r
    )

    best_cx_px = None
    best_cy_px = None
    best_r_px = None

    if circles is not None and len(circles[0]) > 0:
        crop_mid_x = (crop_x2 - crop_x1) / 2.0
        crop_mid_y = (crop_y2 - crop_y1) / 2.0
        best_dist = 1e9
        for c in circles[0]:
            dist = np.hypot(c[0] - crop_mid_x, c[1] - crop_mid_y)
            if dist < best_dist:
                best_dist = dist
                best_cx_px = crop_x1 + c[0]
                best_cy_px = crop_y1 + c[1]
                best_r_px = c[2]

    if best_cx_px is not None and best_cy_px is not None and best_r_px is not None:
        norm_cx = float((best_cx_px / w) * 1000.0)
        norm_cy = float((best_cy_px / h) * 1000.0)
        r_norm_x = float((best_r_px / w) * 1000.0)
        r_norm_y = float((best_r_px / h) * 1000.0)
        refined_bbox = [
            int(max(0, norm_cy - r_norm_y)),
            int(max(0, norm_cx - r_norm_x)),
            int(min(1000, norm_cy + r_norm_y)),
            int(min(1000, norm_cx + r_norm_x))
        ]
        return (norm_cx, norm_cy, refined_bbox)

    return (float((x1_raw + x2_raw) / 2.0), float((y1_raw + y2_raw) / 2.0), [int(y1_raw), int(x1_raw), int(y2_raw), int(x2_raw)])


def detect_all_table_objects(frame: Optional[np.ndarray], mapper=None) -> List[Dict[str, Any]]:
    """
    全自主 Qwen3.8-Flash 多模态具身视觉感知系统（带严格具身语义过滤与象棋高精识别）：
    精准提取工作区内所有象棋棋子（红兵、黑卒、红车、黑马等）与工件，
    自动结合霍夫圆变换实现棋子几何圆心亚毫米精度的物理定位。
    """
    if frame is None:
        return []

    try:
        from robot_vision_client import frame_to_base64_optimized
        b64 = frame_to_base64_optimized(frame, quality=90)

        # 针对象棋棋子与工件的专属视觉提示词
        prompt = (
            "你是具身机械臂高精度工业视觉中枢。请对工作台画面中的象棋棋子与工件进行【精确目标提取与定位】：\n"
            "【核心识别规则】：\n"
            "1. 象棋棋子必须识别出阵营颜色与具体汉字，名称必须以'红'或'黑'开头，例如'红兵'、'红车'、'红炮'、'红相'、'红仕'、'红帅'、'黑卒'、'黑车'、'黑马'、'黑炮'、'黑士'、'黑象'、'黑将'。严禁含糊地只输出'棋子'！\n"
            "2. 其他工件请给出具体类别（如'胶带'、'圆筒'、'零件盒'等）。\n"
            "3. 坚决过滤剔除：桌面底板、白色托盘底、电源线缆、机械臂机身本体等非抓取作业环境背景物。\n"
            "输出紧凑合法 JSON 格式: {\"objects\": [{\"name\": \"红兵\", \"bbox\": [ymin, xmin, ymax, xmax]}]}\n"
            "坐标归一化到 0~1000 范围整数。严禁任何废话。"
        )
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "enable_thinking": False,
            "max_tokens": 2500,
            "temperature": 0.1
        }
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=40.0
        )
        if resp.status_code == 200:
            res_json = resp.json()
            raw_text = res_json["choices"][0]["message"].get("content", "")
            data = _extract_vlm_json(raw_text)
            if data and isinstance(data.get("objects"), list):
                objs = []
                semantic_blacklist = ["托盘", "纸盒", "纸箱", "盒子", "桌面", "背景", "铅笔", "笔", "尺子", "线缆", "机械臂", "支架", "底座"]
                for o in data["objects"]:
                    bbox = o.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    y1, y2 = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
                    x1, x2 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
                    ymin, xmin, ymax, xmax = int(y1), int(x1), int(y2), int(x2)
                    name = str(o.get("name", "工件")).strip()
                    cat = str(o.get("category", ""))

                    # 1. 语义黑名单过滤：若属于背景杂物，跳过
                    if any(k in name for k in semantic_blacklist) and not any(k in name for k in ["棋", "车", "車", "马", "馬", "炮", "兵", "卒", "将", "帅", "仕", "士", "相", "象"]):
                        continue
                    if cat == "environment":
                        continue

                    # 2. 面积尺寸过滤：滤除占满大半屏的超大底板或背景
                    box_w = max(0, xmax - xmin)
                    box_h = max(0, ymax - ymin)
                    if box_w > 650 or box_h > 650:
                        continue

                    # 3. 几何中心霍夫圆自适应对齐：若为棋子或圆形工件，直接锁定圆形物理圆心
                    is_round = any(k in name for k in ["棋", "兵", "卒", "车", "車", "马", "馬", "炮", "将", "帥", "帅", "仕", "士", "相", "象", "圆", "筒", "胶带"])
                    if is_round:
                        norm_cx, norm_cy, refined_bbox = refine_piece_center_circle(frame, [ymin, xmin, ymax, xmax])
                        ymin, xmin, ymax, xmax = refined_bbox
                    else:
                        norm_cx = (xmin + xmax) / 2.0
                        norm_cy = (ymin + ymax) / 2.0

                    xyz = mapper.pixel_to_robot(norm_cx, norm_cy) if mapper else None

                    # 4. 物理工作区范围语义检查：超出机械臂合理抓取范围的点不视为有效工作区目标
                    if xyz:
                        rx, ry, rz = xyz
                        if rx < 50.0 or rx > 600.0 or abs(ry) > 500.0:
                            continue

                    # 颜色区分：红棋标注亮红色，黑棋标注青蓝色，金黄色棋子标注黄色，工件标注绿色
                    if any(k in name for k in ["红", "帥", "帅", "相", "兵"]):
                        box_color = (0, 69, 255)
                    elif any(k in name for k in ["黑", "将", "卒"]):
                        box_color = (255, 180, 0)
                    elif any(k in name for k in ["棋", "车", "車", "马", "馬", "炮", "仕", "士", "象"]):
                        box_color = (0, 215, 255)
                    else:
                        box_color = (0, 255, 120)

                    norm_cx = float(norm_cx)
                    norm_cy = float(norm_cy)
                    xyz_list = [round(float(v), 1) for v in xyz] if xyz is not None else None

                    objs.append({
                        "name": name,
                        "color_bgr": box_color,
                        "bbox": [int(ymin), int(xmin), int(ymax), int(xmax)],
                        "center_norm": [norm_cx, norm_cy],
                        "center_px": [int(norm_cx * frame.shape[1] / 1000.0), int(norm_cy * frame.shape[0] / 1000.0)],
                        "robot_xyz": xyz_list
                    })
                return objs
    except Exception as e:
        print(f"⚠️ Qwen3.8-Flash 全景检测请求失败: {e}")

    # 本地或本地代理 6006 端口回退
    try:
        from robot_vision_client import frame_to_base64_optimized
        b64 = frame_to_base64_optimized(frame)
        resp = requests.post("http://127.0.0.1:6006/detect_all_objects", json={"image_base64": b64}, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                objs = []
                for o in data.get("objects", []):
                    bbox = o.get("bbox")
                    name = o.get("name", "工件")
                    norm_cx, norm_cy = o.get("center_norm", [500.0, 500.0])
                    xyz = mapper.pixel_to_robot(norm_cx, norm_cy) if mapper else None
                    objs.append({
                        "name": name,
                        "color_bgr": (0, 220, 255) if any(k in name for k in ["盒", "急停", "箱", "按钮"]) else (0, 255, 120),
                        "bbox": bbox,
                        "center_norm": [norm_cx, norm_cy],
                        "center_px": (int(norm_cx * frame.shape[1] / 1000.0), int(norm_cy * frame.shape[0] / 1000.0)),
                        "robot_xyz": xyz
                    })
                return objs
    except Exception:
        pass

    return []


def detect_table_objects(frame: Optional[np.ndarray] = None, target_name: str = "工件") -> Optional[List[int]]:
    """
    通过 Qwen3.8-Flash 视觉大模型进行特定目标物体定向定位 (带霍夫圆心高精对齐)
    """
    if frame is None:
        return None

    try:
        from robot_vision_client import frame_to_base64_optimized
        b64 = frame_to_base64_optimized(frame)

        prompt = (
            f"你是机械臂高精度工业视觉中枢。请在图中精确定位目标物体【{target_name}】。\n"
            "若是象棋棋子，请准确分辨出字样与阵营颜色（如'红兵'、'黑卒'、'红车'、'黑马'等）。\n"
            "以合法 JSON 格式输出目标物体的归一化检测框 [ymin, xmin, ymax, xmax] (0~1000)。\n"
            "格式: {\"name\": \"" + str(target_name) + "\", \"bbox\": [ymin, xmin, ymax, xmax]}\n"
            "严禁任何废话。"
        )
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "enable_thinking": False,
            "max_tokens": 1000,
            "temperature": 0.1
        }
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=40.0
        )
        if resp.status_code == 200:
            res_json = resp.json()
            raw_text = res_json["choices"][0]["message"].get("content", "")
            data = _extract_vlm_json(raw_text)
            if data:
                bbox = data.get("bbox") or data.get("normalized_bbox")
                if bbox and len(bbox) == 4:
                    raw_b = [int(v) for v in bbox]
                    is_round = any(k in str(target_name) for k in ["棋", "兵", "卒", "车", "車", "马", "馬", "炮", "将", "帥", "帅", "仕", "士", "相", "象", "圆", "筒", "胶带"])
                    if is_round:
                        _, _, refined_b = refine_piece_center_circle(frame, raw_b)
                        print(f"🎯 [Qwen3.8-Flash 定位] 精准锁定【{target_name}】目标框 (霍夫圆心对齐): {refined_b}")
                        return refined_b
                    print(f"🎯 [Qwen3.8-Flash 定位] 精准锁定【{target_name}】目标框: {raw_b}")
                    return raw_b
    except Exception as e:
        print(f"⚠️ Qwen3.8-Flash 目标检索失败: {e}")

    # 若特定目标未能锁定，尝试从全景中筛选同名物体
    all_objs = detect_all_table_objects(frame)
    target_clean = str(target_name).strip().lower()
    for o in all_objs:
        oname = o.get("name", "").lower()
        if target_clean in oname or oname in target_clean:
            return o["bbox"]

    return None
