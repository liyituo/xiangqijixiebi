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


def is_valid_vector(data: Any, min_length: int = 6) -> bool:
    """
    数据防崩溃校验：严格校验返回数据是否为合法的 6 维数值向量。
    防御 pymycobot 在超时或总线忙时返回 -1 或 None 导致的 len() 崩溃问题。
    """
    if not isinstance(data, (list, tuple)):
        return False
    if len(data) < min_length:
        return False
    return all(isinstance(x, (int, float)) for x in data[:min_length])


def clamp_angles(angles: List[float]) -> List[float]:
    """角度极限钳位，防止超出 Pro 450 硬件限位触发异常"""
    clamped = []
    for i, a in enumerate(angles[:6]):
        min_v = PRO450_LIMITS["angles_min"][i]
        max_v = PRO450_LIMITS["angles_max"][i]
        clamped.append(round(max(min_v, min(max_v, float(a))), 2))
    return clamped


def clamp_coords(coords: List[float], safety_z_min: float = 20.0) -> List[float]:
    """笛卡尔坐标极限钳位，含桌面防撞保护"""
    clamped = []
    for i, c in enumerate(coords[:6]):
        min_v = PRO450_LIMITS["coords_min"][i]
        max_v = PRO450_LIMITS["coords_max"][i]
        val = float(c)
        if i == 2:  # Z 轴桌面安全防护
            val = max(safety_z_min, val)
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

        # 尝试连接
        self.connect()

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
    # 核心功能 2: 状态读取 (带防崩重试与校验)
    # --------------------------------------------------------------------------
    def get_angles(self) -> List[float]:
        """读取 6 轴关节角度 [J1, J2, J3, J4, J5, J6] (带重试防崩与线程锁)"""
        with self.lock:
            if self.is_connected and self.client:
                for _ in range(3):
                    try:
                        raw = self.client.get_angles()
                        if is_valid_vector(raw):
                            angles = [round(float(x), 2) for x in raw[:6]]
                            self.virtual_angles = list(angles)
                            return angles
                    except Exception:
                        pass
                    time.sleep(0.02)
            return list(self.virtual_angles)

    def get_coords(self) -> List[float]:
        """读取末端笛卡尔坐标 [X, Y, Z, Rx, Ry, Rz] (带重试防崩与线程锁)"""
        with self.lock:
            if self.is_connected and self.client:
                for _ in range(3):
                    try:
                        raw = self.client.get_coords()
                        if is_valid_vector(raw):
                            coords = [round(float(x), 2) for x in raw[:6]]
                            self.virtual_coords = list(coords)
                            return coords
                    except Exception:
                        pass
                    time.sleep(0.02)
            return list(self.virtual_coords)

    # --------------------------------------------------------------------------
    # 核心功能 3: 运动驱动 (带极限钳位与参数自适应)
    # --------------------------------------------------------------------------
    def _run_pro450_motion(self, target: List[float], speed: int, coords: bool):
        """异步发送并有界等待到位，失败时终止后续工序。调用方持有 self.lock。"""
        if self.stop_requested.is_set():
            raise RuntimeError("已请求急停，请检查机械臂并重新启动控制程序")

        if coords and hasattr(self.client, "solve_inv_kinematics"):
            angles = self.client.get_angles()
            if not is_valid_vector(angles):
                raise RuntimeError("无法读取当前关节角度，已取消移动")
            solution = self.client.solve_inv_kinematics(target, list(angles[:6]))
            if not is_valid_vector(solution):
                # 触发自适应多维姿态/高度搜索：微调偏航角 Rz、俯仰 Rx、翻滚 Ry 或平滑调整 Z 高度寻求可行逆解，杜绝硬报错
                found_alt = False
                for delta_z in [0.0, -10.0, -20.0, -35.0, 10.0, -50.0]:
                    for alt_rz in [46.0, 0.0, -45.0, 45.0, 90.0, -90.0, 180.0]:
                        for alt_rx in [-178.5, -180.0, -175.0, 180.0]:
                            for alt_ry in [-3.1, 0.0, -5.0, 5.0]:
                                test_target = list(target)
                                test_target[2] = max(PRO450_LIMITS["table_safety_z"] + 5.0, test_target[2] + delta_z)
                                test_target[3] = alt_rx
                                test_target[4] = alt_ry
                                test_target[5] = alt_rz
                                alt_sol = self.client.solve_inv_kinematics(test_target, list(angles[:6]))
                                if is_valid_vector(alt_sol):
                                    print(f"   🔄 [逆解自适应成功] 姿态自适应补偿: Z={test_target[2]:.1f}, Rx={alt_rx}, Ry={alt_ry}, Rz={alt_rz}")
                                    target = test_target
                                    solution = alt_sol
                                    found_alt = True
                                    break
                            if found_alt:
                                break
                        if found_alt:
                            break
                    if found_alt:
                        break
                if not found_alt:
                    raise RuntimeError("目标位置和姿态无有效逆解，已取消移动")

        if self.stop_requested.is_set():
            raise RuntimeError("已请求急停，已取消移动")
        try:
            command = self.client.send_coords if coords else self.client.send_angles
            result = command(target, speed, _async=True)
            if result != 1:
                raise RuntimeError("机械臂未接受移动指令")
            started = time.monotonic()
            idle_since = None
            invalid_count = 0
            while time.monotonic() - started < 45.0:
                if self.stop_requested.is_set():
                    raise RuntimeError("动作已被停止")
                arrived = self.client.is_in_position(target, 1 if coords else 0)
                if arrived == 1:
                    return
                moving = self.client.is_moving()
                if arrived not in (0, 1) or moving not in (0, 1):
                    invalid_count += 1
                    if invalid_count >= 5:
                        raise RuntimeError("无法读取运动状态，请检查机械臂连接")
                else:
                    invalid_count = 0
                # 容差就位检查（由于物理微小误差或法兰重力，硬件内部 is_in_position 判定极其严苛）
                if coords:
                    curr_c = self.get_coords()
                    if is_valid_vector(curr_c) and len(curr_c) >= 3:
                        dist_xyz = ((curr_c[0] - target[0])**2 + (curr_c[1] - target[1])**2 + (curr_c[2] - target[2])**2)**0.5
                        if dist_xyz < 4.0: # XYZ 误差在 4mm 允许物理接触公差内直接判定到达
                            return
                else:
                    curr_a = self.get_angles()
                    if is_valid_vector(curr_a) and len(curr_a) >= 6:
                        max_deg_diff = max(abs(curr_a[j] - target[j]) for j in range(6))
                        if max_deg_diff < 1.5:
                            return

                now = time.monotonic()
                if now - started >= 3.0 and moving == 0:
                    idle_since = idle_since or now
                    if now - idle_since >= 1.0:
                        # 机械臂停止运动，再次核对实际位置误差
                        if coords:
                            curr_c = self.get_coords()
                            if is_valid_vector(curr_c) and len(curr_c) >= 3:
                                dist_xyz = ((curr_c[0] - target[0])**2 + (curr_c[1] - target[1])**2 + (curr_c[2] - target[2])**2)**0.5
                                if dist_xyz < 12.0: # 接近目标点已静止则认可就位
                                    return
                        else:
                            curr_a = self.get_angles()
                            if is_valid_vector(curr_a) and len(curr_a) >= 6:
                                if max(abs(curr_a[j] - target[j]) for j in range(6)) < 3.0:
                                    return
                        raise RuntimeError("机械臂未到达目标且已停止，请检查姿态或可达范围")
                else:
                    idle_since = None
                time.sleep(0.2)
            raise RuntimeError("机械臂移动超时，请检查目标位置和姿态")
        except Exception as motion_error:
            if not self.stop_requested.is_set():
                try:
                    if self.client.stop() != 1:
                        raise RuntimeError("停止请求未获确认")
                except Exception as stop_error:
                    raise RuntimeError(f"{motion_error}；停止请求未获确认，请使用实体急停") from stop_error
            raise

    def send_angles(self, angles: List[float], speed: int = 15):
        """驱动 6 轴关节到指定角度 (带安全限位钳位)"""
        clamped = clamp_angles(angles)
        speed = max(1, min(100, int(speed)))

        with self.lock:
            if self.is_connected and self.client:
                try:
                    if self.stop_requested.is_set():
                        raise RuntimeError("已请求急停，请检查机械臂并重新启动控制程序")
                    if self.is_free_mode:
                        self.disable_free_drag()
                    if self.driver_type == "Pro450Client":
                        self._run_pro450_motion(clamped, speed, coords=False)
                    else:
                        self.client.send_angles(clamped, speed)
                        time.sleep(0.6)
                    self.virtual_angles = list(clamped)
                except Exception as e:
                    print(f"   ❌ send_angles 错误: {e}")
                    raise
            else:
                self.virtual_angles = list(clamped)
                print(f"   🤖 [仿真动作] 关节角度驱动 -> {clamped} | 速度: {speed}")
                time.sleep(0.2)

    def send_coords(self, coords: List[float], speed: int = 15):
        """驱动末端笛卡尔坐标 (智能处理 Pro450Client 与通用版签名不一致)"""
        clamped = clamp_coords(coords, safety_z_min=PRO450_LIMITS["table_safety_z"])
        speed = max(1, min(100, int(speed)))

        with self.lock:
            if self.is_connected and self.client:
                try:
                    if self.stop_requested.is_set():
                        raise RuntimeError("已请求急停，请检查机械臂并重新启动控制程序")
                    if self.is_free_mode:
                        self.disable_free_drag()
                    if self.driver_type == "Pro450Client":
                        self._run_pro450_motion(clamped, speed, coords=True)
                    else:
                        sig = inspect.signature(self.client.send_coords)
                        if "mode" in sig.parameters:
                            self.client.send_coords(clamped, speed, 1)
                        else:
                            self.client.send_coords(clamped, speed)
                        time.sleep(0.6)
                    self.virtual_coords = list(clamped)
                except Exception as e:
                    print(f"   ❌ send_coords 错误: {e}")
                    raise
            else:
                self.virtual_coords = list(clamped)
                print(f"   🤖 [仿真动作] 笛卡尔坐标驱动 -> X={clamped[0]:.1f}, Y={clamped[1]:.1f}, Z={clamped[2]:.1f} | 速度: {speed}")
                time.sleep(0.2)

    # --------------------------------------------------------------------------
    # 核心功能 4: 夹爪精细控制 (全面兼容 Pro 夹爪与标准夹爪)
    # --------------------------------------------------------------------------
    def init_gripper(self, gripper_id: int = 14) -> bool:
        """初始化大象机器人官方自适应电动夹爪 Pro 版 (Modbus 485 总线型, ID=14, 波特率 115200)"""
        with self.lock:
            if self.is_connected and self.client:
                try:
                    self.gripper_id = gripper_id
                    if hasattr(self.client, "set_pro_gripper_init"):
                        print(f"   🔧 正在初始化 Pro 电动夹爪 (Modbus ID: {gripper_id})...")
                        self.client.set_pro_gripper_init(gripper_id)
                        time.sleep(0.3)
                    if hasattr(self.client, "set_pro_gripper_enabled"):
                        self.client.set_pro_gripper_enabled(1, gripper_id)
                        time.sleep(0.2)
                    if hasattr(self.client, "get_pro_gripper_angle"):
                        ang = self.client.get_pro_gripper_angle(gripper_id)
                        if isinstance(ang, (int, float)) and 0 <= ang <= 100:
                            self.virtual_gripper = int(ang)
                    print(f"   🤏 [Pro 自适应夹爪] 初始化完毕！当前实测开度: {self.virtual_gripper}%")
                    return True
                except Exception as e:
                    print(f"   ⚠️ [Pro 自适应夹爪] 初始化异常: {e}")
                    return False
            return False

    def set_gripper(self, action: str, speed: int = 50):
        """离散控制夹爪开合 (OPEN: 15% / CLOSE: 5%)"""
        is_open = (str(action).upper() == "OPEN")
        val = 15 if is_open else 5
        self.set_gripper_value(val, speed)

    def set_gripper_value(self, value: int, speed: int = 50):
        """连续行程百分比控制夹爪 (0~100)"""
        val = max(0, min(100, int(value)))
        self.virtual_gripper = val

        with self.lock:
            if self.is_connected and self.client:
                # 1. Pro450Client 专属 Pro 夹爪方法 (精准传递 Modbus ID 14)
                if hasattr(self.client, "set_pro_gripper_angle"):
                    try:
                        self.client.set_pro_gripper_angle(val, self.gripper_id)
                        time.sleep(0.4)
                        return
                    except TypeError:
                        try:
                            self.client.set_pro_gripper_angle(val)
                            time.sleep(0.4)
                            return
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"   ⚠️ set_pro_gripper_angle 失败: {e}")

                # 2. 通用 set_gripper_value
                if hasattr(self.client, "set_gripper_value"):
                    try:
                        self.client.set_gripper_value(val, speed)
                        time.sleep(0.4)
                        return
                    except Exception:
                        pass

                # 3. 通用 set_gripper_state
                if hasattr(self.client, "set_gripper_state"):
                    try:
                        state = 0 if val > 40 else 1
                        self.client.set_gripper_state(state, speed)
                        time.sleep(0.4)
                        return
                    except Exception:
                        pass
            else:
                tag = "【张开 100%】" if val > 50 else f"【开度 {val}%】" if val > 0 else "【夹紧 0%】"
                print(f"   🤏 [自适应夹爪] -> {tag}")
                time.sleep(0.2)

    def get_gripper_value(self) -> int:
        """获取当前夹爪开度 (0~100)"""
        with self.lock:
            if self.is_connected and self.client:
                # 1. Pro 夹爪 (Modbus ID 14)
                if hasattr(self.client, "get_pro_gripper_angle"):
                    try:
                        ang = self.client.get_pro_gripper_angle(self.gripper_id)
                        if isinstance(ang, (int, float)) and 0 <= ang <= 100:
                            self.virtual_gripper = int(ang)
                            return int(ang)
                    except TypeError:
                        try:
                            ang = self.client.get_pro_gripper_angle()
                            if isinstance(ang, (int, float)) and 0 <= ang <= 100:
                                self.virtual_gripper = int(ang)
                                return int(ang)
                        except Exception:
                            pass
                    except Exception:
                        pass
                # 2. 通用夹爪
                if hasattr(self.client, "get_gripper_value"):
                    try:
                        val = self.client.get_gripper_value()
                        if isinstance(val, (int, float)) and 0 <= val <= 100:
                            self.virtual_gripper = int(val)
                            return int(val)
                    except Exception:
                        pass

            return self.virtual_gripper

    # --------------------------------------------------------------------------
    # 核心功能 5: 头部 RGB LED 与紧急制动
    # --------------------------------------------------------------------------
    def set_color(self, r: int, g: int, b: int):
        """设置末端 RGB 指示灯"""
        self.virtual_led = (r, g, b)
        with self.lock:
            if self.is_connected and self.client:
                try:
                    if hasattr(self.client, "set_color"):
                        self.client.set_color(r, g, b)
                except Exception:
                    pass

    def emergency_stop(self):
        """发送软件停止请求，不等待正在执行动作持有的驱动锁。"""
        self.stop_requested.set()
        client = self.client
        if self.is_connected and client:
            try:
                if hasattr(client, "stop"):
                    result = client.stop()
                    if self.driver_type == "Pro450Client" and result != 1:
                        raise RuntimeError("机械臂未确认停止请求")
                elif hasattr(client, "pause"):
                    client.pause()
                else:
                    raise RuntimeError("当前驱动不支持软件停止")
            except Exception as e:
                print(f"   ⚠️ 停止请求发送失败，请使用实体急停: {e}")
                raise
        print("   🛑 已发送停止请求，请确认机械臂已停止。")
