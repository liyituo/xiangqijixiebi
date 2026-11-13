# -*- coding: utf-8 -*-
"""
大象机器人 x 多模态视觉大模型 (Qwen2.5-VL Turbo) 实时控制中枢
优化亮点：
1. 【免按键全自动拍照】：输入指令即自动从摄像头提取最新一帧发送，无需手动按空格；
2. 【毫秒级轻量传输】：图像在端侧自动压缩为最佳分辨率 (448x448)，网络传输仅需约 20ms；
3. 【实时目标框 HUD 渲染】：大模型返回的物体与坐标，直接在摄像头画面上画绿色框并标出机械臂毫米坐标；
4. 【11 步工业安全闭环】：自动经过空中安全检查点 (ACT-06) 悬停挂起防拖地。
"""

import sys
import time
import json
import base64
import threading
import cv2
import requests
import numpy as np
from typing import Dict, Any, List, Optional

from vision_calibration import HandEyeMapper
from vision_tools import VisionActionExecutor, PhysicalSafetyGuard
from pro450_driver import MyCobotPro450Driver, is_valid_vector, clamp_angles, clamp_coords

# 云服务器多模态 API 接口 (通过 SSH 隧道映射到本地 6006 端口)
CLOUD_VLM_URL = "http://127.0.0.1:6006/robot_vision_command"

# 大象机器人网络配置 (网线连接)
ROBOT_IP = "192.168.0.232"
ROBOT_PORT = 4500

class EnhancedRobotDriver:
    """兼具角度控制、笛卡尔空间点动、夹爪精细调谐与状态反馈的工业驱动器 (myCobot Pro 450 优化版)"""

    def __init__(self, ip: str = ROBOT_IP, port: int = ROBOT_PORT):
        self.ip = ip
        self.port = port
        self.backend = MyCobotPro450Driver(ip=self.ip, port=self.port, auto_scan=True)
        self.mc = self.backend.client
        self.is_connected = self.backend.is_connected
        self.virtual_coords = self.backend.virtual_coords
        self.virtual_angles = self.backend.virtual_angles
        self.virtual_gripper = self.backend.virtual_gripper
        self.virtual_led = self.backend.virtual_led

    def get_coords(self) -> List[float]:
        """获取末端笛卡尔坐标 [x, y, z, rx, ry, rz]"""
        return self.backend.get_coords()

    def get_angles(self) -> List[float]:
        """获取各关节角度 [j1, j2, j3, j4, j5, j6]"""
        return self.backend.get_angles()

    def get_gripper_value(self) -> int:
        """获取当前夹爪开度 (0~100)"""
        return self.backend.get_gripper_value()

    def send_coords(self, coords: List[float], speed: int = 15):
        """驱动机械臂末端到达笛卡尔坐标 [x, y, z, rx, ry, rz]"""
        self.backend.send_coords(coords, speed)

    def send_angles(self, angles: List[float], speed: int = 15):
        """驱动机械臂 6 个关节到指定角度"""
        self.backend.send_angles(angles, speed)

    def init_gripper(self, gripper_id: int = 14) -> bool:
        """初始化 Pro 自适应电动夹爪"""
        return self.backend.init_gripper(gripper_id)

    def set_gripper(self, action: str, speed: int = 50):
        """离散控制夹爪开合 (OPEN/CLOSE)"""
        self.backend.set_gripper(action, speed)

    def set_gripper_value(self, value: int, speed: int = 50):
        """连续百分比控制夹爪开度 (0~100)"""
        self.backend.set_gripper_value(value, speed)

    def set_color(self, r: int, g: int, b: int):
        """设置机械臂头部 RGB 指示灯"""
        self.backend.set_color(r, g, b)

    def release_all_servos(self):
        """释放所有关节舵机扭矩 / 降低阻尼 (自由拖拽示教)"""
        self.backend.enable_free_drag()

    def focus_all_servos(self):
        """锁定所有关节舵机扭矩 (恢复上电刚性保持)"""
        self.backend.disable_free_drag()

    def emergency_stop(self):
        """硬件急停制动"""
        self.backend.emergency_stop()


class CameraStreamThread:
    """常驻后台的低延迟视频流捕获器"""

    def __init__(self, cam_id: int = 0):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.current_frame = None
        self.is_running = True
        self.lock = threading.Lock()
        
        # 启动多线程连续读取
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
            time.sleep(0.01)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def release(self):
        self.is_running = False
        self.cap.release()


def frame_to_base64_optimized(frame: np.ndarray, max_dim: Optional[int] = None, quality: int = 98) -> str:
    """
    全量原生画质 Base64 编码器 (云端大模型直传)：
    默认 100% 保留摄像头原始全分辨率，不作任何裁剪与尺寸压缩，
    采用 98 超高保真画质输出，让云端 Qwen3.8-Flash 获取最清晰的文字笔锋与棋盘交点。
    """
    frame_out = frame
    if max_dim is not None:
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            frame_out = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', frame_out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def main():
    print("=" * 70)
    print("  👁️ 大象机器人 x Qwen2.5-VL 极速多模态控制中枢 (Turbo)")
    print("=" * 70)

    # 1. 启动模块
    mapper = HandEyeMapper()
    driver = EnhancedRobotDriver(ROBOT_IP, ROBOT_PORT)
    executor = VisionActionExecutor(driver, mapper)

    # 2. 启动摄像头后台流
    cam = CameraStreamThread(0)
    time.sleep(1.0)
    if cam.get_latest_frame() is None:
        print("❌ 未检测到摄像头，请检查 USB 接口！")
        return

    print("✅ 摄像头后台常驻流已就绪！")
    print("\n💡 【全新体验】：你不需要再按空格拍照！")
    print("   直接在终端输入指令（例如: '把桌上的红积木抓到P3', '复位', '夹爪张开'），")
    print("   系统将【自动在回车瞬间截取最新画面】并毫秒级送达云端 4080！\n")

    # 全局 HUD 标注变量
    hud_state = {
        "box": None,
        "label": "",
        "xyz": None,
        "status": "Ready. Type instruction to execute."
    }

    # 后台线程渲染 OpenCV HUD 窗口
    def hud_render_loop():
        cv2.namedWindow("FlexArm Vision Agent HUD", cv2.WINDOW_AUTOSIZE)
        while cam.is_running:
            frame = cam.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            disp = frame.copy()

            # 准星
            cv2.drawMarker(disp, (w//2, h//2), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

            # 目标检测框与坐标
            if hud_state["box"]:
                ymin, xmin, ymax, xmax = hud_state["box"]
                px1, py1 = int(xmin * w / 1000.0), int(ymin * h / 1000.0)
                px2, py2 = int(xmax * w / 1000.0), int(ymax * h / 1000.0)
                cv2.rectangle(disp, (px1, py1), (px2, py2), (0, 255, 0), 2)
                lbl = hud_state["label"]
                if hud_state["xyz"]:
                    lbl += f" (X:{hud_state['xyz'][0]}, Y:{hud_state['xyz'][1]})"
                cv2.putText(disp, lbl, (px1, max(25, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 状态栏
            cv2.rectangle(disp, (0, 0), (w, 35), (35, 35, 35), -1)
            cv2.putText(disp, f"STATUS: {hud_state['status']}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("FlexArm Vision Agent HUD", disp)
            if cv2.waitKey(20) & 0xFF == 27:
                cam.is_running = False
                break
        cv2.destroyAllWindows()

    hud_thread = threading.Thread(target=hud_render_loop, daemon=True)
    hud_thread.start()

    # 主控制台交互循环
    while cam.is_running:
        try:
            user_input = input("🗣️ 请输入机械臂控制指令 > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit"]:
                cam.release()
                break

            # 自动截取当前最新一帧
            latest_frame = cam.get_latest_frame()
            if latest_frame is None:
                print("⚠️ 无法获取摄像头画面。")
                continue

            b64_img = frame_to_base64_optimized(latest_frame)
            hud_state["status"] = f"Reasoning: {user_input}..."
            print("🧠 自动捕获当前最新画面，云端 4080 极速多模态推理中...")

            t0 = time.time()
            resp = requests.post(
                CLOUD_VLM_URL,
                json={"instruction": user_input, "image_base64": b64_img},
                timeout=12
            )
            cost_ms = (time.time() - t0) * 1000

            if resp.status_code != 200:
                print(f"❌ 请求失败: {resp.status_code}")
                hud_state["status"] = "Request failed."
                continue

            data = resp.json()
            model_ms = data.get("elapsed_ms", 0)
            print(f"⚡ 大模型响应完成 (模型计算: {model_ms:.1f}ms | 网络往返: {cost_ms:.1f}ms)")

            has_tool = data.get("has_tool_call", False)
            tool_name = data.get("tool_name")
            args = data.get("arguments", {})

            if has_tool and tool_name:
                print(f"🎯 触发工具调用: 【{tool_name}】 -> 参数: {args}")

                if tool_name == "flexarm_pick_and_place":
                    target_obj = args.get("target_object", "工件")
                    bbox = args.get("normalized_bbox", [500, 500, 600, 600])
                    station = args.get("target_station", "P3")

                    # 实时渲染 HUD 框与坐标
                    hud_state["box"] = bbox
                    hud_state["label"] = target_obj
                    hud_state["xyz"] = mapper.bbox_to_robot_center(bbox)
                    hud_state["status"] = f"Moving: {target_obj} -> {station}"

                    # 执行 11 步抓取闭环
                    res = executor.execute_visual_pick_and_place(target_obj, bbox, station)
                    hud_state["status"] = "Pick & Place Success!" if res.get("success") else "Task failed!"

                elif tool_name == "reset_arm":
                    driver.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], speed=15)
                    hud_state["status"] = "Arm reset Home."
                elif tool_name == "wave_hand":
                    print("👋 挥手动作中...")
                    driver.send_angles([0.0, 20.0, -20.0, 0.0, 30.0, 0.0], 20)
                    driver.send_angles([0.0, 20.0, -20.0, 0.0, -30.0, 0.0], 20)
                    driver.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 15)
                    hud_state["status"] = "Wave done."
                elif tool_name == "flexarm_control_gripper":
                    driver.set_gripper(args.get("action", "OPEN"))
                    hud_state["status"] = f"Gripper {args.get('action')}"
                elif tool_name == "flexarm_emergency_stop":
                    driver.emergency_stop()
                    hud_state["status"] = "EMERGENCY STOPPED!"
            else:
                print(f"💬 模型回复: {data.get('visual_description')}")
                hud_state["status"] = "Reply received."

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 运行异常: {e}")

    cam.release()
    print("程序已退出。")

if __name__ == "__main__":
    main()
