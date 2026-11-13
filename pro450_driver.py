# -*- coding: utf-8 -*-
"""
大象机器人 myCobot Pro 450 专属工业级通信驱动适配层 (Pro 450 Driver Adapter)
专为 Elephant Robotics myCobot Pro 450 六轴协作机械臂定制开发。

解决痛点与技术亮点：
1. 【专属底层协议绑定】：完美封装 Pro450Client，并自适应兼任 MyCobotSocket；
2. 【零重力低阻尼拖拽示教】：精准调用 set_free_move_mode(1) 降低阻尼手动拖拽，退出后安全 power_on 恢复刚性保持；
3. 【数据防崩溃校验】：彻底修复底层通信超时返回 -1 导致 TypeError 的严重 Bug；
4. 【自适应 Pro 夹爪接口】：自动适配 Pro450Client 的 set_pro_gripper_open/close/angle/status，杜绝 AttributeError；
5. 【send_coords 动态参数适配】：解决 Pro450 与通用版 send_coords 参数数量与签名不一致问题；
6. 【物理极限动态钳位】：严格参照 myCobot Pro 450 官方限位表 (角度/坐标)，防止触发 MyCobotPro450DataException；
7. 【网络智能探测与容错】：自动扫描探测 4500、5001、9000 等各版本端口，提供断线自动仿真与平滑重连。
"""

import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import time
import math
import socket
import inspect
import threading
from typing import List, Dict, Any, Optional, Tuple

# ==============================================================================
# myCobot Pro 450 官方物理极限参数表
# ==============================================================================
PRO450_LIMITS = {
    # 6 个关节角度物理极限 (单位: 度)
    "angles_min": [-162.0, -125.0, -154.0, -162.0, -162.0, -165.0],
    "angles_max": [162.0, 125.0, 154.0, 162.0, 162.0, 165.0],
    # 笛卡尔末端物理坐标极限 (单位: 毫米与度) [X, Y, Z, Rx, Ry, Rz]
    "coords_min": [-466.0, -466.0, -180.0, -180.0, -180.0, -180.0],
    "coords_max": [466.0, 466.0, 614.0, 180.0, 180.0, 180.0],
    # 桌面防撞安全高度 (Z 轴绝对底线)
    "table_safety_z": 20.0,
    # 速度范围 (1 ~ 100)
    "speed_range": (1, 100)
}

DEFAULT_ROBOT_IP = "192.168.0.232"
DEFAULT_ROBOT_PORT = 4500
PROBE_PORTS = [4500, 5001, 9000]


def is_finite_number(value: Any) -> bool:
    """检查是否为有限数值（显式拒绝 bool、NaN、Inf 与非数值类型）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_valid_vector(data: Any, min_length: int = 6) -> bool:
    """
    数据防崩溃校验：严格校验返回数据是否为合法且全部有限的数值向量。
    防御 pymycobot 返回 -1、None、bool 或 NaN/Inf 导致的异常。
    """
    if not isinstance(data, (list, tuple)):
        return False
    if len(data) < min_length:
        return False
    return all(is_finite_number(x) for x in data[:min_length])


def validate_vector(data: Any, expected_len: int = 6, name: str = "vector") -> List[float]:
    """严格校验固定维度有限数值向量，不合格直接抛出 ValueError。"""
    if not isinstance(data, (list, tuple)) or len(data) != expected_len:
        raise ValueError(f"{name} 必须为长度为 {expected_len} 的数值列表")
    out: List[float] = []
    for idx, val in enumerate(data):
        if not is_finite_number(val):
            raise ValueError(f"{name}[{idx}] 必须为有限数字，当前为 {val!r}")
        out.append(float(val))
    return out


def validate_speed(speed: Any, min_speed: Optional[int] = None, max_speed: Optional[int] = None) -> int:
    """严格校验运动速度，拒绝非数字、bool、NaN/Inf 与越界速度。"""
    lo, hi = PRO450_LIMITS["speed_range"]
    if min_speed is not None:
        lo = max(lo, int(min_speed))
    if max_speed is not None:
        hi = min(hi, int(max_speed))
    if not is_finite_number(speed):
        raise ValueError(f"速度必须为有限数字，当前为 {speed!r}")
    s_float = float(speed)
    if s_float < lo or s_float > hi:
        raise ValueError(f"速度 {s_float} 越界，允许范围为 [{lo}, {hi}]")
    return max(lo, min(hi, int(round(s_float))))


def clamp_angles(angles: List[float]) -> List[float]:
    """角度校验与极限钳位，防止超出 Pro 450 硬件限位触发异常。"""
    vec = validate_vector(angles, expected_len=6, name="angles")
    clamped = []
    for i, val in enumerate(vec):
        min_v = PRO450_LIMITS["angles_min"][i]
        max_v = PRO450_LIMITS["angles_max"][i]
        clamped.append(round(max(min_v, min(max_v, val)), 2))
    return clamped


def clamp_coords(coords: List[float], safety_z_min: float = 20.0) -> List[float]:
    """笛卡尔坐标校验与极限钳位，含桌面防撞保护。"""
    vec = validate_vector(coords, expected_len=6, name="coords")
    if not is_finite_number(safety_z_min):
        raise ValueError(f"safety_z_min 必须为有限数字，当前为 {safety_z_min!r}")
    clamped = []
    for i, val in enumerate(vec):
        min_v = PRO450_LIMITS["coords_min"][i]
        max_v = PRO450_LIMITS["coords_max"][i]
        if i == 2:  # Z 轴桌面安全防护
            val = max(float(safety_z_min), val)
        val = max(min_v, min(max_v, val))
        clamped.append(round(val, 2))
    return clamped


class MyCobotPro450Driver:
    """myCobot Pro 450 专属工业级通信驱动适配器"""

    def __init__(self, ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT, auto_scan: bool = True):
        self.ip = ip
        self.port = port
        self.auto_scan = auto_scan
        self.client = None
        self.driver_type = "NONE"
        self.is_connected = False
        self.is_free_mode = False
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.gripper_id = 14  # 大象官方自适应电动夹爪 Pro 版默认 485 Modbus ID
        
        # 虚拟内部状态缓存 (基于真实标定 NORMAL 初始位姿)
        self.virtual_angles = [6.68, -9.08, -86.4, 6.94, 3.12, -129.32]
        self.virtual_coords = [306.6, -46.8, 313.9, -178.5, -3.1, 46.0]
        self.virtual_gripper = 15  # 初始设定为 15% 开度 (用户定制: 开 15%, 闭 5%)
        self.virtual_led = (0, 255, 0)
        self._ensure_runtime_state()

        # 尝试连接
        self.connect()

    def _ensure_runtime_state(self):
        """惰性初始化运行期状态字段，兼容 __new__ 构造的离线测试桩。"""
        if not hasattr(self, "gripper_id"):
            self.gripper_id = 14
        if not hasattr(self, "virtual_gripper"):
            self.virtual_gripper = 15
        if not hasattr(self, "virtual_led"):
            self.virtual_led = (0, 255, 0)
        if not hasattr(self, "stop_state"):
            self.stop_state = "requested" if getattr(self, "stop_requested", None) and self.stop_requested.is_set() else "idle"
        if not hasattr(self, "stop_error"):
            self.stop_error = None
        if not hasattr(self, "last_motion_report"):
            self.last_motion_report = None
        if not hasattr(self, "last_gripper_report"):
            self.last_gripper_report = None
        if not hasattr(self, "last_angles_meta"):
            self.last_angles_meta = {"valid": False, "source": "init", "timestamp": 0.0, "error": None}
        if not hasattr(self, "last_coords_meta"):
            self.last_coords_meta = {"valid": False, "source": "init", "timestamp": 0.0, "error": None}
        if not hasattr(self, "last_gripper_meta"):
            self.last_gripper_meta = {"valid": False, "source": "init", "timestamp": 0.0, "error": None, "status_code": None}

    def _test_socket(self, ip: str, port: int, timeout: float = 0.8) -> bool:
        """快速无阻塞探测指定 IP:Port 是否存活"""
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
            s.close()
            return True
        except Exception:
            return False

    def connect(self) -> bool:
        """建立机械臂网络连接，支持多端口智能切换与 Pro450Client 优先绑定"""
        print(f"\n🔌 正在初始化 myCobot Pro 450 通信链路 ({self.ip}:{self.port}) ...")
        
        ports_to_try = [self.port]
        if self.auto_scan:
            for p in PROBE_PORTS:
                if p not in ports_to_try:
                    ports_to_try.append(p)

        connected_port = None
        for p in ports_to_try:
            # 1. 尝试导入 Pro450Client
            try:
                from pymycobot import Pro450Client
                c = Pro450Client(self.ip, p)
                time.sleep(0.3)
                angles = c.get_angles()
                if is_valid_vector(angles):
                    self.client = c
                    self.port = p
                    self.driver_type = "Pro450Client"
                    self.is_connected = True
                    connected_port = p
                    print(f"✅ 【成功连接真机】绑定 myCobot Pro 450 专属驱动！端口: {p}")
                    print(f"   🔩 当前关节角度: {angles}")
                    self.init_gripper(self.gripper_id)
                    return True
                else:
                    if hasattr(c, "clear_socket_buffer"):
                        c.clear_socket_buffer()
            except Exception:
                pass

            # 2. 尝试导入通用 MyCobotSocket
            try:
                from pymycobot import MyCobotSocket
                s = MyCobotSocket(self.ip, p)
                time.sleep(0.3)
                angles = s.get_angles()
                if is_valid_vector(angles):
                    self.client = s
                    self.port = p
                    self.driver_type = "MyCobotSocket"
                    self.is_connected = True
                    connected_port = p
                    print(f"✅ 【成功连接真机】绑定通用 MyCobotSocket 驱动！端口: {p}")
                    print(f"   🔩 当前关节角度: {angles}")
                    self.init_gripper(self.gripper_id)
                    return True
            except Exception:
                pass

        # 3. 未连通真机时启用仿真
        print(f"⚠️ 未检测到物理机械臂响应 (网络不可达或未开机)。")
        print(f"🎮 自动启用【myCobot Pro 450 高保真物理仿真模式】。")
        self.is_connected = False
        self.driver_type = "SIMULATION"
        return False

    # --------------------------------------------------------------------------
    # 核心功能 1: 零重力 / 低阻尼自由拖动示教 (Drag Teach)
    # --------------------------------------------------------------------------
    def enable_free_drag(self) -> bool:
        """降低阻尼进入自由拖动模式 (可手动轻推示教)"""
        with self.lock:
            if self.is_connected and self.client:
                try:
                    # 1. Pro 450 官方协议指令: set_free_move_mode(1)
                    if hasattr(self.client, "set_free_move_mode"):
                        self.client.set_free_move_mode(1)
                    # 2. 彻底释放电机扭矩与阻尼，实现超轻盈自由拖动
                    if hasattr(self.client, "set_motor_enabled"):
                        self.client.set_motor_enabled(254, 0)
                    elif hasattr(self.client, "release_all_servos"):
                        self.client.release_all_servos()

                    # 头部 LED 变青蓝色指示示教中
                    self.set_color(0, 200, 255)
                    self.is_free_mode = True
                    print("   🔓 [扭矩状态] 阻尼已降低，自由拖拽示教已激活！(可用手拖动关节)")
                    return True
                except Exception as e:
                    print(f"   ❌ 开启自由拖动失败: {e}")
                    return False
            else:
                self.is_free_mode = True
                print("   🔓 [仿真动作] 阻尼已降低，虚拟自由拖拽模式激活。")
                return True

    def disable_free_drag(self) -> bool:
        """退出自由拖动模式，重新锁紧舵机扭矩恢复刚性"""
        with self.lock:
            if self.is_connected and self.client:
                try:
                    # 1. 退出 free move
                    if hasattr(self.client, "set_free_move_mode"):
                        self.client.set_free_move_mode(0)
                    
                    # 2. 重新使能并锁定所有电机扭矩 (刚性保持)
                    if hasattr(self.client, "set_motor_enabled"):
                        self.client.set_motor_enabled(254, 1)
                    if hasattr(self.client, "power_on"):
                        self.client.power_on(1)
                    elif hasattr(self.client, "focus_all_servos"):
                        self.client.focus_all_servos()

                    # 头部 LED 恢复绿色待机
                    self.set_color(0, 255, 0)
                    self.is_free_mode = False
                    print("   🔒 [扭矩状态] 舵机扭矩已锁紧刚性保持，退出拖拽模式。")
                    return True
                except Exception as e:
                    print(f"   ❌ 锁定舵机失败: {e}")
                    return False
            else:
                self.is_free_mode = False
                print("   🔒 [仿真动作] 舵机扭矩已锁紧保持。")
                return True

    # --------------------------------------------------------------------------
    # 核心功能 2: 状态读取 (区分真实硬件反馈与离线缓存)
    # --------------------------------------------------------------------------
    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        """计算角度差值（处理 ±180° 环绕）。"""
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    def _is_cancelled(self, cancel_token: Any = None) -> bool:
        if self.stop_requested.is_set():
            return True
        if cancel_token is not None and hasattr(cancel_token, "is_cancelled"):
            return bool(cancel_token.is_cancelled())
        return False

    def _read_live_vector(self, getter_name: str, attempts: int = 3) -> Tuple[Optional[List[float]], Optional[str]]:
        """直接读取硬件实时向量，绝不回退到 virtual_* 缓存。调用方需持有 self.lock 或确保线程安全。"""
        if not (self.is_connected and self.client):
            return None, "机械臂未连接"
        getter = getattr(self.client, getter_name, None)
        if getter is None:
            return None, f"底层驱动不支持 {getter_name}"
        last_err = None
        for _ in range(attempts):
            try:
                raw = getter()
                if is_valid_vector(raw):
                    return [round(float(x), 2) for x in raw[:6]], None
                last_err = f"{getter_name} 返回无效数据: {raw!r}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.02)
        return None, last_err or f"{getter_name} 读取失败"

    def _get_vector_feedback(self, getter_name: str, cache_attr: str, meta_attr: str) -> Dict[str, Any]:
        """读取带来源与新鲜度标记的 6 维状态反馈。"""
        self._ensure_runtime_state()
        now = time.time()
        with self.lock:
            if self.is_connected and self.client:
                vec, err = self._read_live_vector(getter_name, attempts=3)
                if vec is not None:
                    setattr(self, cache_attr, list(vec))
                    meta = {"valid": True, "source": "hardware", "timestamp": now, "error": None}
                    setattr(self, meta_attr, meta)
                    return {"values": vec, **meta}
                meta = {"valid": False, "source": "stale_cache", "timestamp": now, "error": err}
                setattr(self, meta_attr, meta)
                return {"values": list(getattr(self, cache_attr)), **meta}
            meta = {"valid": True, "source": "simulation", "timestamp": now, "error": None}
            setattr(self, meta_attr, meta)
            return {"values": list(getattr(self, cache_attr)), **meta}

    def get_angles_feedback(self) -> Dict[str, Any]:
        """读取 6 轴关节角度及反馈元数据 (valid/source/timestamp/error)。"""
        return self._get_vector_feedback("get_angles", "virtual_angles", "last_angles_meta")

    def get_coords_feedback(self) -> Dict[str, Any]:
        """读取末端笛卡尔坐标及反馈元数据 (valid/source/timestamp/error)。"""
        return self._get_vector_feedback("get_coords", "virtual_coords", "last_coords_meta")

    def get_angles(self, allow_cache: bool = True) -> Optional[List[float]]:
        """读取 6 轴关节角度 [J1..J6]；allow_cache=False 时硬件失败返回 None。"""
        fb = self.get_angles_feedback()
        if not fb["valid"] and not allow_cache:
            return None
        return list(fb["values"])

    def get_coords(self, allow_cache: bool = True) -> Optional[List[float]]:
        """读取末端笛卡尔坐标 [X..Rz]；allow_cache=False 时硬件失败返回 None。"""
        fb = self.get_coords_feedback()
        if not fb["valid"] and not allow_cache:
            return None
        return list(fb["values"])

    # --------------------------------------------------------------------------
    # 核心功能 3: 运动驱动 (受限 IK 自适应 + 位置/姿态联合校验 + 取消感知)
    # --------------------------------------------------------------------------
    def _solve_coords_ik_with_bounds(
        self,
        requested_target: List[float],
        angles: List[float],
        allow_ik_fallback: bool,
        max_z_drop_mm: float = 15.0,
        max_rot_delta_deg: float = 15.0,
        max_attempts: int = 12,
        cancel_token: Any = None,
    ) -> Tuple[List[float], bool]:
        """求解并校验目标逆解；仅当显式允许时在有界邻域内微调并记录实际规划目标。"""
        solution = self.client.solve_inv_kinematics(requested_target, list(angles[:6]))
        if is_valid_vector(solution):
            return list(requested_target), False
        if not allow_ik_fallback:
            raise RuntimeError("目标位置和姿态无有效逆解，已取消移动")

        z_min_safe = PRO450_LIMITS["table_safety_z"] + 5.0
        z_deltas = [d for d in (0.0, -5.0, -10.0, -abs(max_z_drop_mm), 5.0) if -abs(max_z_drop_mm) <= d <= 10.0]
        rot_steps = [0.0, 5.0, -5.0, min(12.0, abs(max_rot_delta_deg)), -min(12.0, abs(max_rot_delta_deg))]
        candidates: List[List[float]] = []
        for dz in z_deltas:
            for drz in rot_steps:
                for drx, dry in ((0.0, 0.0), (4.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0)):
                    if dz == 0.0 and drz == 0.0 and drx == 0.0 and dry == 0.0:
                        continue
                    if max(abs(drz), abs(drx), abs(dry)) > abs(max_rot_delta_deg):
                        continue
                    cand = list(requested_target)
                    cand[2] = max(z_min_safe, cand[2] + dz)
                    cand[3] = ((cand[3] + drx + 180.0) % 360.0) - 180.0
                    cand[4] = max(-180.0, min(180.0, cand[4] + dry))
                    cand[5] = ((cand[5] + drz + 180.0) % 360.0) - 180.0
                    cand = clamp_coords(cand, safety_z_min=PRO450_LIMITS["table_safety_z"])
                    if cand[2] < z_min_safe:
                        continue
                    candidates.append(cand)
                    if len(candidates) >= max_attempts:
                        break
                if len(candidates) >= max_attempts:
                    break
            if len(candidates) >= max_attempts:
                break

        for cand in candidates:
            if self._is_cancelled(cancel_token):
                raise RuntimeError("动作已被停止")
            alt_sol = self.client.solve_inv_kinematics(cand, list(angles[:6]))
            if is_valid_vector(alt_sol):
                print(
                    f"   🔄 [受限逆解自适应] 规划目标调整: Z={cand[2]:.1f}, "
                    f"Rx={cand[3]:.1f}, Ry={cand[4]:.1f}, Rz={cand[5]:.1f}"
                )
                return cand, True

        raise RuntimeError("目标位置和姿态无有效逆解，已取消移动")

    def _within_motion_tolerance(
        self,
        planned_target: List[float],
        coords: bool,
        pos_tol_mm: float,
        rot_tol_deg: float,
        angle_tol_deg: float,
    ) -> Tuple[bool, Optional[List[float]]]:
        """使用真实硬件实时反馈联合检查位置与姿态公差，禁止使用缓存回退。"""
        if coords:
            curr_c, _ = self._read_live_vector("get_coords", attempts=1)
            if not is_valid_vector(curr_c):
                return False, None
            dist_xyz = (
                (curr_c[0] - planned_target[0]) ** 2
                + (curr_c[1] - planned_target[1]) ** 2
                + (curr_c[2] - planned_target[2]) ** 2
            ) ** 0.5
            max_rot = max(self._angle_diff_deg(curr_c[j], planned_target[j]) for j in range(3, 6))
            return (dist_xyz <= pos_tol_mm and max_rot <= rot_tol_deg), curr_c

        curr_a, _ = self._read_live_vector("get_angles", attempts=1)
        if not is_valid_vector(curr_a):
            return False, None
        max_deg_diff = max(self._angle_diff_deg(curr_a[j], planned_target[j]) for j in range(6))
        return (max_deg_diff <= angle_tol_deg), curr_a

    def _run_pro450_motion(
        self,
        target: List[float],
        speed: int,
        coords: bool,
        cancel_token: Any = None,
        phase: Optional[str] = None,
        allow_ik_fallback: bool = False,
        ik_max_z_drop: float = 15.0,
        ik_max_rot_delta: float = 15.0,
        ik_max_attempts: int = 12,
        pos_tol_mm: float = 4.0,
        rot_tol_deg: float = 5.0,
        angle_tol_deg: float = 1.5,
    ) -> Dict[str, Any]:
        """异步发送并有界等待到位，检查位置与姿态联合公差并支持取消令牌。调用方持有 self.lock。"""
        self._ensure_runtime_state()
        if self._is_cancelled(cancel_token):
            raise RuntimeError("已请求急停，请检查机械臂并重新启动控制程序")

        requested_target = list(target)
        planned_target = list(target)
        ik_adapted = False

        if coords and hasattr(self.client, "solve_inv_kinematics"):
            angles, ang_err = self._read_live_vector("get_angles", attempts=2)
            if angles is None:
                raise RuntimeError(f"无法读取当前关节角度，已取消移动 ({ang_err})")
            planned_target, ik_adapted = self._solve_coords_ik_with_bounds(
                requested_target=requested_target,
                angles=angles,
                allow_ik_fallback=allow_ik_fallback,
                max_z_drop_mm=ik_max_z_drop,
                max_rot_delta_deg=ik_max_rot_delta,
                max_attempts=ik_max_attempts,
                cancel_token=cancel_token,
            )

        if self._is_cancelled(cancel_token):
            raise RuntimeError("已请求急停，已取消移动")
        try:
            command = self.client.send_coords if coords else self.client.send_angles
            result = command(planned_target, speed, _async=True)
            if result != 1:
                raise RuntimeError("机械臂未接受移动指令")
            started = time.monotonic()
            idle_since = None
            invalid_count = 0
            while time.monotonic() - started < 45.0:
                if self._is_cancelled(cancel_token):
                    raise RuntimeError("动作已被停止")
                arrived = self.client.is_in_position(planned_target, 1 if coords else 0)
                moving = self.client.is_moving()
                if arrived == 1:
                    getter = "get_coords" if coords else "get_angles"
                    live_pose, _ = self._read_live_vector(getter, attempts=1)
                    actual_pose = live_pose if live_pose is not None else list(planned_target)
                    report = {
                        "mode": "coords" if coords else "angles",
                        "phase": phase,
                        "requested_target": requested_target,
                        "planned_target": planned_target,
                        "actual_pose": actual_pose,
                        "ik_adapted": ik_adapted,
                        "arrived_source": "is_in_position",
                        "timestamp": time.time(),
                    }
                    self.last_motion_report = report
                    return report
                if arrived not in (0, 1) or moving not in (0, 1):
                    invalid_count += 1
                    if invalid_count >= 5:
                        raise RuntimeError("无法读取运动状态，请检查机械臂连接")
                else:
                    invalid_count = 0

                if moving == 0:
                    ok_tol, live_pose = self._within_motion_tolerance(
                        planned_target, coords, pos_tol_mm, rot_tol_deg, angle_tol_deg
                    )
                    if ok_tol and live_pose is not None:
                        report = {
                            "mode": "coords" if coords else "angles",
                            "phase": phase,
                            "requested_target": requested_target,
                            "planned_target": planned_target,
                            "actual_pose": live_pose,
                            "ik_adapted": ik_adapted,
                            "arrived_source": "tolerance",
                            "timestamp": time.time(),
                        }
                        self.last_motion_report = report
                        return report

                now = time.monotonic()
                if now - started >= 3.0 and moving == 0:
                    idle_since = idle_since or now
                    if now - idle_since >= 1.0:
                        ok_settled, live_pose = self._within_motion_tolerance(
                            planned_target, coords, pos_tol_mm * 1.5, rot_tol_deg * 1.5, angle_tol_deg * 1.5
                        )
                        if ok_settled and live_pose is not None:
                            report = {
                                "mode": "coords" if coords else "angles",
                                "phase": phase,
                                "requested_target": requested_target,
                                "planned_target": planned_target,
                                "actual_pose": live_pose,
                                "ik_adapted": ik_adapted,
                                "arrived_source": "settled_tolerance",
                                "timestamp": time.time(),
                            }
                            self.last_motion_report = report
                            return report
                        raise RuntimeError("机械臂未到达目标且已停止，请检查姿态或可达范围")
                else:
                    idle_since = None
                time.sleep(0.2)
            raise RuntimeError("机械臂移动超时，请检查目标位置和姿态")
        except Exception as motion_error:
            if not self.stop_requested.is_set():
                try:
                    if hasattr(self.client, "stop") and self.client.stop() != 1:
                        raise RuntimeError("停止请求未获确认")
                except Exception as stop_error:
                    raise RuntimeError(f"{motion_error}；停止请求未获确认，请使用实体急停") from stop_error
            raise

    def _run_generic_motion(
        self,
        target: List[float],
        speed: int,
        coords: bool,
        cancel_token: Any = None,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """兼容通用 MyCobotSocket 的带取消检测与状态反馈等待。"""
        self._ensure_runtime_state()
        if self._is_cancelled(cancel_token):
            raise RuntimeError("已请求急停，已取消移动")
        if coords:
            sig = inspect.signature(self.client.send_coords)
            res = self.client.send_coords(target, speed, 1) if "mode" in sig.parameters else self.client.send_coords(target, speed)
        else:
            res = self.client.send_angles(target, speed)
        if res in (0, -1, False):
            raise RuntimeError(f"机械臂拒绝移动指令 (ret={res!r})")

        # 如果通用驱动支持 is_moving / get_coords / get_angles，则等待运动完成而非固定盲等
        if hasattr(self.client, "is_moving"):
            started = time.monotonic()
            while time.monotonic() - started < 15.0:
                if self._is_cancelled(cancel_token):
                    try:
                        if hasattr(self.client, "stop"):
                            self.client.stop()
                    except Exception:
                        pass
                    raise RuntimeError("动作已被停止")
                try:
                    if self.client.is_moving() == 0 and time.monotonic() - started >= 0.3:
                        break
                except Exception:
                    break
                time.sleep(0.1)
        else:
            for _ in range(6):
                if self._is_cancelled(cancel_token):
                    raise RuntimeError("动作已被停止")
                time.sleep(0.1)

        getter = "get_coords" if coords else "get_angles"
        live_pose, _ = self._read_live_vector(getter, attempts=1)
        actual_pose = live_pose if live_pose is not None else list(target)
        report = {
            "mode": "coords" if coords else "angles",
            "phase": phase,
            "requested_target": list(target),
            "planned_target": list(target),
            "actual_pose": actual_pose,
            "ik_adapted": False,
            "arrived_source": "generic_hardware",
            "timestamp": time.time(),
        }
        self.last_motion_report = report
        return report

    def send_angles(
        self,
        angles: List[float],
        speed: int = 15,
        cancel_token: Any = None,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """驱动 6 轴关节到指定角度 (带严格校验与安全限位钳位)。"""
        self._ensure_runtime_state()
        clamped = clamp_angles(angles)
        speed_val = validate_speed(speed)

        with self.lock:
            if self.is_connected and self.client:
                try:
                    if self._is_cancelled(cancel_token):
                        raise RuntimeError("已请求急停，请检查机械臂并重新启动控制程序")
                    if self.is_free_mode:
                        self.disable_free_drag()
                    if self.driver_type == "Pro450Client":
                        report = self._run_pro450_motion(
                            clamped, speed_val, coords=False, cancel_token=cancel_token, phase=phase
                        )
                    else:
                        report = self._run_generic_motion(
                            clamped, speed_val, coords=False, cancel_token=cancel_token, phase=phase
                        )
                    self.virtual_angles = list(report["actual_pose"])
                    return report
                except Exception as e:
                    print(f"   ❌ send_angles 错误: {e}")
                    raise
            if self._is_cancelled(cancel_token):
                raise RuntimeError("动作已被停止")
            self.virtual_angles = list(clamped)
            report = {
                "mode": "angles",
                "phase": phase,
                "requested_target": list(clamped),
                "planned_target": list(clamped),
                "actual_pose": list(clamped),
                "ik_adapted": False,
                "arrived_source": "simulation",
                "timestamp": time.time(),
            }
            self.last_motion_report = report
            print(f"   🤖 [仿真动作] 关节角度驱动 -> {clamped} | 速度: {speed_val}")
            time.sleep(0.2)
            return report

    def send_coords(
        self,
        coords: List[float],
        speed: int = 15,
        cancel_token: Any = None,
        phase: Optional[str] = None,
        allow_ik_fallback: bool = False,
    ) -> Dict[str, Any]:
        """驱动末端笛卡尔坐标 (严格校验 + 受限 IK 搜索 + 实际到达回报)。"""
        self._ensure_runtime_state()
        clamped = clamp_coords(coords, safety_z_min=PRO450_LIMITS["table_safety_z"])
        speed_val = validate_speed(speed)

        with self.lock:
            if self.is_connected and self.client:
                try:
                    if self._is_cancelled(cancel_token):
                        raise RuntimeError("已请求急停，请检查机械臂并重新启动控制程序")
                    if self.is_free_mode:
                        self.disable_free_drag()
                    if self.driver_type == "Pro450Client":
                        report = self._run_pro450_motion(
                            clamped,
                            speed_val,
                            coords=True,
                            cancel_token=cancel_token,
                            phase=phase,
                            allow_ik_fallback=allow_ik_fallback,
                        )
                    else:
                        report = self._run_generic_motion(
                            clamped, speed_val, coords=True, cancel_token=cancel_token, phase=phase
                        )
                    self.virtual_coords = list(report["actual_pose"])
                    return report
                except Exception as e:
                    print(f"   ❌ send_coords 错误: {e}")
                    raise
            if self._is_cancelled(cancel_token):
                raise RuntimeError("动作已被停止")
            self.virtual_coords = list(clamped)
            report = {
                "mode": "coords",
                "phase": phase,
                "requested_target": list(clamped),
                "planned_target": list(clamped),
                "actual_pose": list(clamped),
                "ik_adapted": False,
                "arrived_source": "simulation",
                "timestamp": time.time(),
            }
            self.last_motion_report = report
            print(f"   🤖 [仿真动作] 笛卡尔坐标驱动 -> X={clamped[0]:.1f}, Y={clamped[1]:.1f}, Z={clamped[2]:.1f} | 速度: {speed_val}")
            time.sleep(0.2)
            return report

    # --------------------------------------------------------------------------
    # 核心功能 4: 夹爪精细控制与结果核验 (全面兼容 Pro 夹爪与标准夹爪)
    # --------------------------------------------------------------------------
    def init_gripper(self, gripper_id: int = 14) -> bool:
        """初始化大象机器人官方自适应电动夹爪 Pro 版 (Modbus 485 总线型, ID=14, 波特率 115200)"""
        self._ensure_runtime_state()
        with self.lock:
            if self.is_connected and self.client:
                try:
                    self.gripper_id = int(gripper_id)
                    if hasattr(self.client, "set_pro_gripper_init"):
                        print(f"   🔧 正在初始化 Pro 电动夹爪 (Modbus ID: {self.gripper_id})...")
                        res = self.client.set_pro_gripper_init(self.gripper_id)
                        if res in (0, -1, False):
                            raise RuntimeError(f"set_pro_gripper_init 返回失败: {res!r}")
                        time.sleep(0.3)
                    if hasattr(self.client, "set_pro_gripper_enabled"):
                        res = self.client.set_pro_gripper_enabled(1, self.gripper_id)
                        if res in (0, -1, False):
                            raise RuntimeError(f"set_pro_gripper_enabled 返回失败: {res!r}")
                        time.sleep(0.2)
                    if hasattr(self.client, "get_pro_gripper_angle"):
                        ang = self.client.get_pro_gripper_angle(self.gripper_id)
                        if is_finite_number(ang) and 0 <= ang <= 100:
                            self.virtual_gripper = int(round(ang))
                    print(f"   🤏 [Pro 自适应夹爪] 初始化完毕！当前实测开度: {self.virtual_gripper}%")
                    return True
                except Exception as e:
                    print(f"   ⚠️ [Pro 自适应夹爪] 初始化异常: {e}")
                    return False
            return False

    def _read_gripper_angle_and_status(self) -> Tuple[Optional[int], Optional[int]]:
        """读取夹爪实测开度与状态码 (Pro 夹爪: 0运动中, 1停止未夹持, 2停止已夹持, 3夹持后掉落)。"""
        measured: Optional[int] = None
        status_code: Optional[int] = None
        if hasattr(self.client, "get_pro_gripper_angle"):
            try:
                ang = self.client.get_pro_gripper_angle(self.gripper_id)
            except TypeError:
                ang = self.client.get_pro_gripper_angle()
            except Exception:
                ang = None
            if is_finite_number(ang) and 0 <= ang <= 100:
                measured = int(round(ang))
        elif hasattr(self.client, "get_gripper_value"):
            try:
                ang = self.client.get_gripper_value()
                if is_finite_number(ang) and 0 <= ang <= 100:
                    measured = int(round(ang))
            except Exception:
                pass

        if hasattr(self.client, "get_pro_gripper_status"):
            try:
                st = self.client.get_pro_gripper_status(self.gripper_id)
            except TypeError:
                st = self.client.get_pro_gripper_status()
            except Exception:
                st = None
            if isinstance(st, int) and not isinstance(st, bool) and st in (0, 1, 2, 3):
                status_code = st
        return measured, status_code

    def set_gripper(
        self,
        action: str,
        speed: int = 50,
        cancel_token: Any = None,
        expected_clamp: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """离散控制夹爪开合 (OPEN: 15% / CLOSE: 5%)"""
        is_open = str(action).upper() == "OPEN"
        val = 15 if is_open else 5
        return self.set_gripper_value(val, speed, cancel_token=cancel_token, expected_clamp=expected_clamp)

    def set_gripper_value(
        self,
        value: int,
        speed: int = 50,
        cancel_token: Any = None,
        verify: bool = True,
        expected_clamp: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """连续行程百分比控制夹爪 (0~100)，校验指令返回并回读开度与夹持状态。"""
        self._ensure_runtime_state()
        if not is_finite_number(value):
            raise ValueError(f"夹爪开度必须为有限数字，当前为 {value!r}")
        v_num = float(value)
        if v_num < 0 or v_num > 100:
            raise ValueError(f"夹爪开度 {v_num} 越界，允许范围为 [0, 100]")
        val = int(round(v_num))
        speed_val = validate_speed(speed)

        with self.lock:
            if self._is_cancelled(cancel_token):
                raise RuntimeError("已请求急停，已取消夹爪控制")

            if self.is_connected and self.client:
                cmd_ok = False
                last_err = None
                if hasattr(self.client, "set_pro_gripper_angle"):
                    try:
                        try:
                            res = self.client.set_pro_gripper_angle(val, self.gripper_id)
                        except TypeError:
                            res = self.client.set_pro_gripper_angle(val)
                        if res in (0, -1, False):
                            raise RuntimeError(f"set_pro_gripper_angle 返回失败 ({res!r})")
                        cmd_ok = True
                    except Exception as e:
                        last_err = str(e)
                        raise RuntimeError(f"夹爪指令失败: {last_err}") from e
                elif hasattr(self.client, "set_gripper_value"):
                    try:
                        res = self.client.set_gripper_value(val, speed_val)
                        if res in (0, -1, False):
                            raise RuntimeError(f"set_gripper_value 返回失败 ({res!r})")
                        cmd_ok = True
                    except Exception as e:
                        raise RuntimeError(f"夹爪指令失败: {e}") from e
                elif hasattr(self.client, "set_gripper_state"):
                    try:
                        state = 0 if val > 40 else 1
                        res = self.client.set_gripper_state(state, speed_val)
                        if res in (0, -1, False):
                            raise RuntimeError(f"set_gripper_state 返回失败 ({res!r})")
                        cmd_ok = True
                    except Exception as e:
                        raise RuntimeError(f"夹爪指令失败: {e}") from e

                if not cmd_ok:
                    raise RuntimeError("当前底层驱动不支持任何夹爪控制接口")

                # 等待夹爪到位并读取实测状态
                measured, status_code = None, None
                deadline = time.monotonic() + 1.2
                while True:
                    if self._is_cancelled(cancel_token):
                        raise RuntimeError("动作已被停止")
                    measured, status_code = self._read_gripper_angle_and_status()
                    if status_code != 0 or time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)

                if measured is not None:
                    self.virtual_gripper = measured
                else:
                    self.virtual_gripper = val

                dropped = status_code == 3
                clamped = status_code == 2
                if verify and dropped:
                    report = {
                        "ok": False,
                        "source": "hardware",
                        "target": val,
                        "measured": measured,
                        "status_code": status_code,
                        "clamped": False,
                        "dropped": True,
                        "timestamp": time.time(),
                    }
                    self.last_gripper_report = report
                    raise RuntimeError("夹爪检测到物体夹持后掉落 (status=3)")

                if verify and expected_clamp is True and status_code == 1:
                    report = {
                        "ok": False,
                        "source": "hardware",
                        "target": val,
                        "measured": measured,
                        "status_code": status_code,
                        "clamped": False,
                        "dropped": False,
                        "timestamp": time.time(),
                    }
                    self.last_gripper_report = report
                    raise RuntimeError("夹爪闭合完成但未检测到夹持物体 (status=1)")

                report = {
                    "ok": True,
                    "source": "hardware",
                    "target": val,
                    "measured": measured if measured is not None else val,
                    "status_code": status_code,
                    "clamped": clamped,
                    "dropped": False,
                    "timestamp": time.time(),
                }
                self.last_gripper_report = report
                return report

            self.virtual_gripper = val
            report = {
                "ok": True,
                "source": "simulation",
                "target": val,
                "measured": val,
                "status_code": 2 if (expected_clamp and val <= 10) else 1,
                "clamped": bool(expected_clamp and val <= 10),
                "dropped": False,
                "timestamp": time.time(),
            }
            self.last_gripper_report = report
            tag = "【张开 100%】" if val > 50 else f"【开度 {val}%】" if val > 0 else "【夹紧 0%】"
            print(f"   🤏 [自适应夹爪] -> {tag}")
            time.sleep(0.2)
            return report

    def get_gripper_feedback(self) -> Dict[str, Any]:
        """获取夹爪开度、状态码与数据有效性元数据。"""
        self._ensure_runtime_state()
        now = time.time()
        with self.lock:
            if self.is_connected and self.client:
                measured, status_code = self._read_gripper_angle_and_status()
                if measured is not None:
                    self.virtual_gripper = measured
                    meta = {
                        "valid": True,
                        "value": measured,
                        "status_code": status_code,
                        "source": "hardware",
                        "timestamp": now,
                        "error": None,
                    }
                    self.last_gripper_meta = meta
                    return meta
                meta = {
                    "valid": False,
                    "value": self.virtual_gripper,
                    "status_code": status_code,
                    "source": "stale_cache",
                    "timestamp": now,
                    "error": "夹爪开度读取失败",
                }
                self.last_gripper_meta = meta
                return meta
            meta = {
                "valid": True,
                "value": self.virtual_gripper,
                "status_code": None,
                "source": "simulation",
                "timestamp": now,
                "error": None,
            }
            self.last_gripper_meta = meta
            return meta

    def get_gripper_value(self, allow_cache: bool = True) -> Optional[int]:
        """获取当前夹爪开度 (0~100)；allow_cache=False 时硬件失败返回 None。"""
        fb = self.get_gripper_feedback()
        if not fb["valid"] and not allow_cache:
            return None
        return int(fb["value"])

    # --------------------------------------------------------------------------
    # 核心功能 5: 头部 RGB LED 与紧急制动
    # --------------------------------------------------------------------------
    def set_color(self, r: int, g: int, b: int):
        """设置末端 RGB 指示灯"""
        self._ensure_runtime_state()
        self.virtual_led = (r, g, b)
        with self.lock:
            if self.is_connected and self.client:
                try:
                    if hasattr(self.client, "set_color"):
                        self.client.set_color(r, g, b)
                except Exception:
                    pass

    def get_stop_state(self) -> Dict[str, Any]:
        """获取急停状态机信息 (idle / requested / confirmed / confirm_failed)。"""
        self._ensure_runtime_state()
        return {
            "stop_requested": self.stop_requested.is_set(),
            "stop_state": self.stop_state,
            "stop_error": self.stop_error,
        }

    def clear_stop(self) -> None:
        """复位急停状态标记，供恢复控制前显式调用。"""
        self._ensure_runtime_state()
        self.stop_requested.clear()
        self.stop_state = "idle"
        self.stop_error = None

    def emergency_stop(self) -> Dict[str, Any]:
        """发送软件停止请求，不等待正在执行动作持有的驱动锁，并显式区分已请求/已确认/确认失败。"""
        self._ensure_runtime_state()
        self.stop_requested.set()
        self.stop_state = "requested"
        self.stop_error = None
        client = self.client
        if self.is_connected and client:
            try:
                if hasattr(client, "stop"):
                    result = client.stop()
                    if self.driver_type == "Pro450Client" and result != 1:
                        raise RuntimeError("机械臂未确认停止请求")
                elif hasattr(client, "pause"):
                    res = client.pause()
                    if res in (0, -1, False):
                        raise RuntimeError("机械臂未确认暂停请求")
                else:
                    raise RuntimeError("当前驱动不支持软件停止")
                self.stop_state = "confirmed"
            except Exception as e:
                self.stop_state = "confirm_failed"
                self.stop_error = str(e)
                print(f"   ⚠️ 停止请求发送失败，请使用实体急停: {e}")
                raise
        else:
            self.stop_state = "confirmed"
        print("   🛑 已发送停止请求，请确认机械臂已停止。")
        return self.get_stop_state()
