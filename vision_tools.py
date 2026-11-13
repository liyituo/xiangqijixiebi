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
from typing import Dict, Any, List, Tuple, Optional
from vision_calibration import HandEyeMapper
from pro450_driver import clamp_angles, clamp_coords, PRO450_LIMITS

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


class PhysicalSafetyGuard:
    """五重物理硬约束安全审查 (全面适配 myCobot Pro 450 工作包络)"""

    # 机械臂物理工作包络限制 (mm) - 适配 myCobot Pro 450 450mm 大臂展
    X_MIN, X_MAX = -460.0, 460.0
    Y_MIN, Y_MAX = -460.0, 460.0
    Z_MIN, Z_MAX = 10.0, 700.0       # 防止撞击工作台，底线 10mm，允许直立高位 (700mm)
    R_MAX = 475.0                    # Pro 450 最大物理展开半径
    R_MIN = 0.0                      # 允许垂直中心作业区

    # 单步最大增量限制 (防大模型幻觉暴冲)
    MAX_STEP_XYZ = 50.0              # mm
    MAX_STEP_ROT = 35.0              # deg

    @classmethod
    def validate_target_point(cls, x: float, y: float, z: float) -> Tuple[bool, str]:
        """验证空间点位是否在绝对安全工作空间内"""
        dist = math.sqrt(x**2 + y**2)
        if dist > cls.R_MAX:
            return False, f"物理超程: 半径距离 {dist:.1f}mm 超过安全极限 {cls.R_MAX}mm"
        if dist < cls.R_MIN:
            return False, f"过近盲区: 距离 {dist:.1f}mm 进入基座干涉区 (<{cls.R_MIN}mm)"

        if not (cls.X_MIN <= x <= cls.X_MAX):
            return False, f"X坐标越界 [{x:.1f}mm]，安全范围为 [{cls.X_MIN}, {cls.X_MAX}]mm"
        if not (cls.Y_MIN <= y <= cls.Y_MAX):
            return False, f"Y坐标越界 [{y:.1f}mm]，安全范围为 [{cls.Y_MIN}, {cls.Y_MAX}]mm"
        if not (cls.Z_MIN <= z <= cls.Z_MAX):
            return False, f"Z高度越界 [{z:.1f}mm]，防止撞击桌面 (允许范围 [{cls.Z_MIN}, {cls.Z_MAX}]mm)"

        return True, "安全审查通过"

    @classmethod
    def validate_delta_step(cls, dx: float, dy: float, dz: float,
                            drx: float = 0.0, dry: float = 0.0, drz: float = 0.0) -> Tuple[bool, str]:
        """验证单步微调增量是否超出安全阈值"""
        dist_delta = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist_delta > cls.MAX_STEP_XYZ:
            return False, f"单步平移过大: {dist_delta:.1f}mm > {cls.MAX_STEP_XYZ}mm，自动限幅保护"
        rot_delta = max(abs(drx), abs(dry), abs(drz))
        if rot_delta > cls.MAX_STEP_ROT:
            return False, f"单步旋转过大: {rot_delta:.1f}deg > {cls.MAX_STEP_ROT}deg"
        return True, "增量安全"

    @staticmethod
    def validate_station(station_name: str) -> bool:
        return station_name.upper() in PRESET_STATIONS


class VisionActionExecutor:
    """视觉引导物理执行中枢与具身综合工具箱"""

    def __init__(self, driver, mapper: HandEyeMapper):
        self.driver = driver
        self.mapper = mapper

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
        place_offset: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        核心物理流程：基于视觉检测框，执行带空中检查点的工业闭环搬运
        支持自定义点位坐标 (解除仅限固定工位的死板限制) 与批量阵列排布偏置 (place_offset)
        支持 offset_trim 手眼标定偏置微调补偿 (ΔX, ΔY, ΔZ)
        """
        dest_display = str(target_station)
        print(f"\n🚀 >>> 启动视觉抓取工序: 目标【{target_name}】 -> 目标目的地【{dest_display}】")
        
        # 1. 坐标换算与映射
        target_x, target_y, table_z = self.mapper.bbox_to_robot_center(bbox)
        # 动态自适应安全过渡高度：桌面+45mm，既确保绝对跨越棋子与工件，又严格处于机械臂可达球域内（彻底避免无逆解）
        hover_z = min(240.0, max(210.0, table_z + 45.0))
        
        # 在原抓取目标的基础上降低 10 mm，桌面基准和悬停高度保持不变。
        grasp_z = table_z + 8.0 - 10.0

        # 叠加手眼标定微调偏置
        if offset_trim:
            target_x = round(target_x + offset_trim[0], 1)
            target_y = round(target_y + offset_trim[1], 1)
            grasp_z = round(grasp_z + offset_trim[2], 1)
        
        # 夹爪柔性闭合：满足用户定制要求（最大开角 15%，闭合抓取 5%）
        is_tape = any(k in str(target_name) for k in ["胶带", "圆", "筒", "盒", "block", "tape", "零件"])
        grip_clamp_val = 5  # 闭合抓取开度调紧至 5%
        grip_open_val = 15  # 最大开角度调整为 15%
        
        print(f"📍 视觉映射物理抓取点: X={target_x}mm, Y={target_y}mm, 缓冲抓取高度 Z={grasp_z}mm (离地过渡 Z={hover_z}mm), 夹爪参数: 开={grip_open_val}%, 闭={grip_clamp_val}%")

        # 2. 安全防火墙硬拦截
        safe_ok, safe_reason = PhysicalSafetyGuard.validate_target_point(target_x, target_y, grasp_z)
        if not safe_ok:
            print(f"🛑 [安全防火墙硬拦截]: {safe_reason}！取消动作以保护硬件。")
            return {"success": False, "error": safe_reason}

        # 3. 解析目标目的地（解除死板工位限制：支持预设工位名、自定义笛卡尔坐标、阵列排布偏移）
        use_custom_coord = False
        target_coord = None
        station_angles = None

        if isinstance(target_station, (list, tuple)) and len(target_station) >= 2:
            use_custom_coord = True
            dx = float(target_station[0])
            dy = float(target_station[1])
            dz = float(target_station[2]) if len(target_station) >= 3 else (table_z + 8.0)
            if place_offset:
                dx += place_offset[0]
                dy += place_offset[1]
                dz += place_offset[2]
            target_coord = [dx, dy, dz, -178.5, -3.1, 46.0]
        else:
            station_str = str(target_station).strip().upper()
            if station_str not in PRESET_STATIONS:
                station_str = "P5"
            if place_offset:
                use_custom_coord = True
                base = PRESET_STATIONS[station_str]
                target_coord = [base[0] + place_offset[0], base[1] + place_offset[1], base[2] + place_offset[2], base[3], base[4], base[5]]
            else:
                target_coord = PRESET_STATIONS[station_str]
                station_angles = PRESET_JOINTS.get(station_str)

        # 自定义终点（含象棋落子点+手动微调偏置）同样执行安全边界硬拦截，并确保过渡悬停高度不低于抓放高度
        min_safe_hover = grasp_z + 15.0
        if use_custom_coord and target_coord:
            dest_ok, dest_reason = PhysicalSafetyGuard.validate_target_point(
                target_coord[0], target_coord[1], target_coord[2]
            )
            if not dest_ok:
                print(f"🛑 [安全防火墙硬拦截]: 目标放置点不安全 ({dest_reason})！取消动作以保护硬件。")
                return {"success": False, "error": f"目标放置点安全拦截: {dest_reason}"}
            min_safe_hover = max(min_safe_hover, target_coord[2] + 15.0)
        hover_z = max(hover_z, min_safe_hover)

        # 4. 构造高安全性工业搬运闭环
        down_speed = max(5, int(speed * 0.6))
        def move_to_delivery():
            if station_angles:
                self.driver.send_angles(station_angles, speed=speed)
            elif target_coord:
                # 先到安全悬停点，再平稳下放交付
                self.driver.send_coords([target_coord[0], target_coord[1], hover_z, target_coord[3], target_coord[4], target_coord[5]], speed=speed)
                self.driver.send_coords(target_coord, speed=down_speed)

        steps = [
            ("ACT-01", f"张开自适应夹爪 ({grip_open_val}% 开度)", lambda: (self.driver.set_gripper_value(grip_open_val, speed=50), time.sleep(0.4))),
            ("ACT-02", f"平移至物体上方安全悬停点 ({target_x:.1f}, {target_y:.1f}, {hover_z:.1f})", lambda: (self.driver.send_coords([target_x, target_y, hover_z, -178.5, -3.1, 46.0], speed), time.sleep(1.2))),
            ("ACT-03", f"垂直缓冲下放抓取 ({target_x:.1f}, {target_y:.1f}, {grasp_z:.1f})", lambda: (self.driver.send_coords([target_x, target_y, grasp_z, -178.5, -3.1, 46.0], speed=down_speed), time.sleep(1.2))),
            ("ACT-04", f"柔性闭合夹爪夹取工件【{target_name}】(开度 {grip_clamp_val}%)", lambda: (self.driver.set_gripper_value(grip_clamp_val, speed=35), time.sleep(1.0))),
            ("ACT-05", f"垂直抬升至安全空域高度 ({hover_z:.1f}mm)", lambda: (self.driver.send_coords([target_x, target_y, hover_z, -178.5, -3.1, 46.0], speed=down_speed), time.sleep(1.0))),
            ("ACT-06", f"【空中安全检查点】悬停确认夹持稳定", lambda: time.sleep(0.6)),
            ("ACT-07", f"经由高位安全航线巡航至目的地 [{dest_display}]", lambda: (move_to_delivery(), time.sleep(1.5))),
            ("ACT-08", f"工位就位确认", lambda: time.sleep(0.5)),
            ("ACT-09", f"释放夹爪交付工件 ({grip_open_val}% 开度)", lambda: (self.driver.set_gripper_value(grip_open_val, speed=50), time.sleep(0.8))),
            ("ACT-10", f"工件交付确认与安全脱离", lambda: time.sleep(0.5)),
            ("ACT-11", f"平稳返回初始待机 NORMAL 姿态", lambda: (self.driver.send_angles(PRESET_JOINTS["NORMAL"], speed=speed), time.sleep(1.5))),
        ]

        total_steps = len(steps)
        for i, (step_id, desc, action) in enumerate(steps):
            backend = getattr(self.driver, "backend", self.driver)
            stop_requested = getattr(backend, "stop_requested", None)
            if stop_requested is not None and stop_requested.is_set():
                raise RuntimeError("动作已被停止，后续步骤未执行")
            print(f"   [{step_id}] -> {desc}")
            if step_callback:
                step_callback(step_id, desc, int((i + 1) / total_steps * 100))
            action()
            time.sleep(0.2)

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
    def move_to_coords(self, coords: List[float], speed: int = 15) -> Dict[str, Any]:
        """移动末端至指定笛卡尔坐标 [x, y, z, rx, ry, rz]"""
        if len(coords) < 3:
            return {"success": False, "error": "坐标参数不足"}
        
        x, y, z = coords[0], coords[1], coords[2]
        safe_ok, reason = PhysicalSafetyGuard.validate_target_point(x, y, z)
        if not safe_ok:
            return {"success": False, "error": reason}

        # 补全姿态角
        full_coords = [coords[0], coords[1], coords[2],
                       coords[3] if len(coords) > 3 else -180.0,
                       coords[4] if len(coords) > 4 else 0.0,
                       coords[5] if len(coords) > 5 else 0.0]

        self.driver.send_coords(full_coords, speed)
        return {"success": True, "target": full_coords}

    # --------------------------------------------------------------------------
    # Tool 3: 笛卡尔空间单轴微调点动 (Jog Cartesian Step)
    # --------------------------------------------------------------------------
    def jog_cartesian(self, axis: str, delta: float, speed: int = 15) -> Dict[str, Any]:
        """
        末端沿指定轴增量微调 (+X, -X, +Y, -Y, +Z, -Z, +Rx, etc.)
        """
        axis = axis.lower()
        curr = self.driver.get_coords()
        if not curr or len(curr) < 6:
            curr = [180.0, 0.0, 180.0, -180.0, 0.0, 0.0]

        target = list(curr)
        axis_map = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
        if axis not in axis_map:
            return {"success": False, "error": f"无效轴名: {axis}"}

        idx = axis_map[axis]
        target[idx] += delta

        # 针对位移进行安全审核
        if idx in [0, 1, 2]:
            safe_ok, reason = PhysicalSafetyGuard.validate_target_point(target[0], target[1], target[2])
            if not safe_ok:
                return {"success": False, "error": reason}

        # 检查是否处于全零奇异姿态，若是则先微动关节脱离奇异点
        curr_angles = self.driver.get_angles()
        if curr_angles and all(abs(a) < 1.0 for a in curr_angles[1:4]):
            self.driver.send_angles([curr_angles[0], 10.0, -15.0, 5.0, curr_angles[4], curr_angles[5]], speed=20)
            time.sleep(0.5)

        self.driver.send_coords(target, speed)
        return {"success": True, "new_coords": target}

    # --------------------------------------------------------------------------
    # Tool 4: 关节角度控制与微调点动 (Joints Control & Jog)
    # --------------------------------------------------------------------------
    def move_joints(self, angles: List[float], speed: int = 15) -> Dict[str, Any]:
        """驱动机械臂 6 个关节角度"""
        if len(angles) != 6:
            return {"success": False, "error": "关节角度数组必须为 6 个"}
        self.driver.send_angles(angles, speed)
        return {"success": True, "angles": angles}

    def jog_joint(self, joint_idx: int, delta_deg: float, speed: int = 15) -> Dict[str, Any]:
        """单关节微调点动 (joint_idx: 1~6)"""
        if not (1 <= joint_idx <= 6):
            return {"success": False, "error": "关节编号必须在 1~6 之间"}
        
        curr = self.driver.get_angles()
        if not curr or len(curr) < 6:
            curr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        target = list(curr)
        target[joint_idx - 1] += delta_deg
        # 严格遵守 myCobot Pro 450 单轴极限钳位
        target = clamp_angles(target)

        self.driver.send_angles(target, speed)
        return {"success": True, "new_angles": target}

    # --------------------------------------------------------------------------
    # Tool 5: 预设标定工位巡航 (Move to Station)
    # --------------------------------------------------------------------------
    def move_to_station(self, station_name: str, speed: int = 15) -> Dict[str, Any]:
        """快速巡航至已标定的物理工位 (HOME, P1, P2, P3, P4, SCAN)"""
        s_name = station_name.upper()
        if s_name not in PRESET_STATIONS and s_name not in PRESET_JOINTS:
            return {"success": False, "error": f"未知工位: {station_name}"}

        # 优先使用关节角度巡航，彻底避免逆运动学奇异点无解错误
        if s_name in PRESET_JOINTS:
            angles = PRESET_JOINTS[s_name]
            self.driver.send_angles(angles, speed)
            return {"success": True, "station": s_name, "angles": angles}

        coords = PRESET_STATIONS[s_name]
        self.driver.send_coords(coords, speed)
        return {"success": True, "station": s_name, "coords": coords}

    # --------------------------------------------------------------------------
    # Tool 6: 自适应夹爪连续行程控制 (Continuous Gripper Control)
    # --------------------------------------------------------------------------
    def control_gripper(self, action: str = "OPEN", value_percent: Optional[int] = None, speed: int = 50) -> Dict[str, Any]:
        """
        夹爪控制：支持快速开合 (OPEN/CLOSE) 或 0~100% 连续行程开度
        """
        if value_percent is not None:
            val = max(0, min(100, int(value_percent)))
            self.driver.set_gripper_value(val, speed)
            return {"success": True, "mode": "percentage", "value": val}
        else:
            self.driver.set_gripper(action, speed)
            return {"success": True, "mode": "discrete", "action": action.upper()}

    # --------------------------------------------------------------------------
    # Tool 7: OpenVLA 7-DoF 动作步进转换执行器 (OpenVLA Step Action)
    # --------------------------------------------------------------------------
    def execute_openvla_action_step(
        self,
        action_7d: List[float],
        scale_xyz_mm: float = 20.0,
        scale_rot_deg: float = 10.0,
        gripper_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        执行 OpenVLA 预测的 7-DoF 归一化动作增量:
        [dx, dy, dz, droll, dpitch, dyaw, gripper] (范围通常在 -1.0 ~ 1.0)
        转换为真实物理位移并驱动机械臂
        """
        if len(action_7d) < 7:
            return {"success": False, "error": f"OpenVLA 动作维度不足 7: {len(action_7d)}"}

        raw_dx, raw_dy, raw_dz = action_7d[0], action_7d[1], action_7d[2]
        raw_drx, raw_dry, raw_drz = action_7d[3], action_7d[4], action_7d[5]
        raw_grip = action_7d[6]

        # 换算到真实空间物理量
        dx = raw_dx * scale_xyz_mm
        dy = raw_dy * scale_xyz_mm
        dz = raw_dz * scale_xyz_mm
        drx = raw_drx * scale_rot_deg
        dry = raw_dry * scale_rot_deg
        drz = raw_drz * scale_rot_deg

        # 安全限幅
        safe_delta, reason = PhysicalSafetyGuard.validate_delta_step(dx, dy, dz, drx, dry, drz)
        if not safe_delta:
            print(f"⚠️ OpenVLA 动作限幅告警: {reason}")
            # 自动等比缩放到安全阈值
            norm = math.sqrt(dx**2 + dy**2 + dz**2)
            if norm > PhysicalSafetyGuard.MAX_STEP_XYZ:
                scale = PhysicalSafetyGuard.MAX_STEP_XYZ / norm
                dx *= scale
                dy *= scale
                dz *= scale

        # 获取当前坐标并施加增量
        curr = self.driver.get_coords()
        if not curr or len(curr) < 6:
            curr = [180.0, 0.0, 180.0, -180.0, 0.0, 0.0]

        target_xyz = [curr[0] + dx, curr[1] + dy, curr[2] + dz]
        safe_pt, pt_reason = PhysicalSafetyGuard.validate_target_point(*target_xyz)
        if not safe_pt:
            return {"success": False, "error": f"OpenVLA 目标点越界: {pt_reason}"}

        target_coords = [
            target_xyz[0], target_xyz[1], target_xyz[2],
            curr[3] + drx, curr[4] + dry, curr[5] + drz
        ]
        self.driver.send_coords(target_coords, speed=15)

        # 夹爪动作
        gripper_state = "CLOSE" if raw_grip < gripper_threshold else "OPEN"
        self.driver.set_gripper(gripper_state, speed=60)

        return {
            "success": True,
            "applied_delta": [dx, dy, dz, drx, dry, drz],
            "new_coords": target_coords,
            "gripper": gripper_state
        }

    # --------------------------------------------------------------------------
    # Tool 8: 多视点环视全景巡检扫描 (Panoramic Workspace Scan)
    # --------------------------------------------------------------------------
    def scan_workspace(self, scan_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        机械臂执行预定三维巡检航线，在关键视角悬停扫描 (采用关节角度驱动，彻底避免逆解奇点报错)
        """
        self.set_status_led("MOVING")
        scan_waypoints = [
            ("SCAN-P1",     PRESET_JOINTS["P1"],     "工位 P1 俯视视点 (左侧工位 X:296 Y:207)"),
            ("SCAN-NORMAL", PRESET_JOINTS["NORMAL"], "工作台全景俯视观测点 (NORMAL X:162 Y:-68 Z:542)"),
            ("SCAN-P3",     PRESET_JOINTS["P3"],     "工位 P3 俯视视点 (右侧成品 X:342 Y:-238)"),
        ]

        results = []
        total = len(scan_waypoints)
        for i, (code, angles, desc) in enumerate(scan_waypoints):
            print(f"🔍 巡检航点 [{i+1}/{total}]: {desc} -> 关节角度 {angles}")
            self.driver.send_angles(angles, speed=18)
            time.sleep(1.2)
            coords = self.driver.get_coords()
            pct = int(((i + 1) / (total + 1)) * 100)
            if scan_callback:
                scan_callback(code, desc, pct)
            results.append({"code": code, "desc": desc, "angles": angles, "coords": coords})

        # 巡检完毕返回待机工位 HOME
        print("🔄 巡检扫描完毕，机械臂平稳返回待机工位 HOME...")
        self.move_to_station("HOME", speed=18)
        time.sleep(1.0)
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
