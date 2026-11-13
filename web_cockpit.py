# -*- coding: utf-8 -*-
"""
大象机器人具身视觉操控大屏与综合工具箱 (FlexArm Web Cockpit & Tool Center)
功能矩阵：
1. 实时 MJPEG 摄像头低延迟画面流 (带目标检测框 HUD 与准星渲染)
2. 画布直接点击抓取 (Click-to-Pick -> 11 步闭环搬运)
3. 多模态大模型自然语言/语音指令解析执行 (Web Speech API)
4. 机械臂全向微调点动控制盘 (Cartesian 6-DoF Jog Pad: ±X, ±Y, ±Z, ±Rx, ±Ry, ±Rz)
5. 6 轴关节独立调谐控制与度数反馈 (Joint Jog Pad: J1~J6)
6. 连续百分比自适应夹爪滑动条 (0% 闭合 ~ 100% 全开)
7. 标定工位一键巡航 (HOME, P1, P2, P3, P4, SCAN)
8. 自由拖拽示教模式开关 (Drag Teach ON/OFF)
9. 头部 RGB LED 状态指示灯切换
10. 全景工作台巡检扫描 (Panoramic Workspace Scan)
11. OpenVLA 7-DoF 动作执行器与模拟接口
12. 毫秒级硬件急停制动 (Emergency Stop)
"""

import os
import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import time
import json
import base64
import threading
import webbrowser
import re
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
import math
import cv2
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template_string, request, jsonify

from env_loader import get_vlm_api_key
from vision_calibration import HandEyeMapper
from vision_tools import VisionActionExecutor, PhysicalSafetyGuard, PRESET_STATIONS, PRESET_JOINTS, detect_table_objects, detect_all_table_objects, refine_piece_center_circle
from robot_vision_client import EnhancedRobotDriver, CameraStreamThread, frame_to_base64_optimized

app = Flask(__name__)

# 配置
DEEPSEEK_API_KEY = get_vlm_api_key("VLM_API_KEY", "DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("VLM_MODEL", os.environ.get("DEEPSEEK_MODEL", "qwen3.8-flash"))
DEEPSEEK_API_URL = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
CLOUD_VLM_URL = "http://127.0.0.1:6006/robot_vision_command"
ROBOT_IP = "192.168.0.232"
ROBOT_PORT = 4500

def call_deepseek_embodied_vlm(instruction: str, frame: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """直接调用 DeepSeek-V4.1-Flash (Vision) 进行多模态自然语言意图理解与目标工件视觉定位"""
    if not DEEPSEEK_API_KEY:
        return None
    try:
        t0 = time.time()
        user_content = []
        if frame is not None:
            b64 = frame_to_base64_optimized(frame)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        user_content.append({"type": "text", "text": instruction})

        system_prompt = (
            "你是一个工业机械臂具身视觉控制中枢。请根据人类指令与工作台画面进行【深度具身语义理解】：\n"
            "【规则】：\n"
            "1. 不要把画面里看到的每一个杂物或背景都当成作业工件。只有人类明确指示要操作的目标，或下棋/分拣的核心目标才执行抓取。\n"
            "2. 深度分析动作语义与目标目的地：\n"
            "   - 若指令为【吃子/清理/移开/丢弃】，目标工位设为 P4（废料/弃子区）；\n"
            "   - 若指令为【抓取/搬运/存放/下棋】，根据语义匹配到合理工位（如人类指定的 P1~P5，默认为 P3 成品区）；\n"
            "   - 必须精准给出目标物体的归一化检测框 [ymin, xmin, ymax, xmax] (0~1000)。\n"
            "3. 输出标准格式：\n"
            "   - 动作指令输出：{\"intent\": \"pick_and_place\", \"has_tool_call\": true, \"tool_name\": \"flexarm_pick_and_place\", "
            "\"arguments\": {\"target_object\": \"真实工件/棋子名\", \"normalized_bbox\": [ymin, xmin, ymax, xmax], \"target_station\": \"工位名称\"}, "
            "\"visual_description\": \"语义执行说明\"}\n"
            "   - 若指令为观察、询问、汇报状态或闲聊，严禁触发抓取，输出：{\"intent\": \"chat\", \"reply\": \"针对画面的详细语义解答\"}\n"
            "只输出合法 JSON，严禁废话。"
        )

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.1
        }

        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=15.0
        )
        if resp.status_code == 200:
            raw_text = resp.json()["choices"][0]["message"].get("content", "")
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                data = json.loads(match.group(0))
                data["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
                return data
    except Exception as e:
        print(f"⚠️ DeepSeek VLM 直接调用异常: {e}")
    return None

def enforce_single_cockpit_instance():
    """彻底终止任何旧的 web_cockpit.py 进程以及占用 5000 端口的进程，防止多实例并发争抢机械臂 Socket 通信信道"""
    try:
        import psutil
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                pid = proc.info['pid']
                if pid == current_pid or pid == os.getppid():
                    continue
                cmd_args = proc.info.get('cmdline') or []
                if any(arg.endswith('web_cockpit.py') for arg in cmd_args):
                    print(f"⚠️ 发现残留的 web_cockpit 进程 (PID: {pid})，正在彻底终止...")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
    
    # 释放可能被占用的 5000 端口
    try:
        import subprocess, re
        out = subprocess.check_output("netstat -ano -p tcp", shell=True).decode()
        current_pid = os.getpid()
        for line in out.strip().splitlines():
            m = re.search(r":5000\s+.*LISTENING\s+(\d+)", line)
            if m:
                pid = int(m.group(1))
                if pid != current_pid and pid > 0:
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                    time.sleep(0.5)
    except Exception:
        pass

enforce_single_cockpit_instance()

# 初始化硬件与算法组件 (全局单例)
mapper = HandEyeMapper()
driver = EnhancedRobotDriver(ROBOT_IP, ROBOT_PORT)
executor = VisionActionExecutor(driver, mapper)

def _sync_calib_offset_locked():
    """若标定配置文件被外部更新，同步磁盘中的手眼微调偏置到遥测字典 (调用前需持有 state_lock)"""
    if mapper.check_reload():
        ox, oy, oz = mapper.get_offset()
        telemetry["calib_offset_x"] = ox
        telemetry["calib_offset_y"] = oy
        telemetry["calib_offset_z"] = oz

def get_calib_offset():
    """实时获取最新手眼微调偏置 (ΔX, ΔY, ΔZ mm)，供象棋对弈抓放调用"""
    with state_lock:
        _sync_calib_offset_locked()
        return (
            round(float(telemetry.get("calib_offset_x", 0.0)), 1),
            round(float(telemetry.get("calib_offset_y", 0.0)), 1),
            round(float(telemetry.get("calib_offset_z", 0.0)), 1),
        )

from xiangqi_agent import XiangqiAgent, XiangqiGridMapper
from xiangqi_engine import INITIAL_FEN
xiangqi_grid_mapper = XiangqiGridMapper(hand_eye_mapper=mapper)
xiangqi_agent = XiangqiAgent(
    driver=driver,
    executor=executor,
    grid_mapper=xiangqi_grid_mapper,
    offset_provider=get_calib_offset,
)
cam = CameraStreamThread(0)

# 轨迹录制数据容器
TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
os.makedirs(TRAJ_DIR, exist_ok=True)
recorded_waypoints: List[Dict[str, Any]] = []

latest_file = os.path.join(TRAJ_DIR, "latest_trajectory.json")
if os.path.exists(latest_file):
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
            if isinstance(_loaded, dict) and "waypoints" in _loaded:
                recorded_waypoints = _loaded["waypoints"]
            elif isinstance(_loaded, list):
                recorded_waypoints = _loaded
    except Exception:
        pass

# 全局可重入状态锁 (防嵌套死锁)
state_lock = threading.RLock()
_init_ox, _init_oy, _init_oz = mapper.get_offset()
telemetry = {
    "cloud_model": "Qwen3.8-Flash",
    "cloud_status": "ONLINE",
    "last_latency_ms": 350.0,
    "robot_status": "CONNECTED" if driver.is_connected else "SIMULATION",
    "motion_speed": 20,
    "current_step": "ACT-00 待机中",
    "step_progress": 0,
    "is_busy": False,
    "calib_offset_x": _init_ox,
    "calib_offset_y": _init_oy,
    "calib_offset_z": _init_oz,
    "live_detected_objects": [],
    "last_detected_obj": "无",
    "last_bbox": None,
    "last_coords": None,
    "drag_teach": False,
    "gripper_val": 15,
    "coords": [162.0, -68.0, 542.0, 168.3, -19.4, 50.0],
    "angles": [5.0, -4.0, -7.0, -57.0, 5.0, -132.0],
    "show_xiangqi_grid": True,
    "logs": [
        f"[{time.strftime('%H:%M:%S')}] FlexArm 具身操控中枢启动就绪",
        f"[{time.strftime('%H:%M:%S')}] 驱动模式: {'真机网口 (192.168.0.232:4500)' if driver.is_connected else '高保真物理仿真模式'}",
        f"[{time.strftime('%H:%M:%S')}] 手眼标定单应性矩阵已挂载"
    ]
}

def add_log(msg: str):
    with state_lock:
        t_str = time.strftime('%H:%M:%S')
        telemetry["logs"].append(f"[{t_str}] {msg}")
        if len(telemetry["logs"]) > 60:
            telemetry["logs"].pop(0)

def pose_telemetry_loop():
    """
    全局唯一定时遥测守护线程 (稳态 2Hz 采样):
    - 坚决杜绝在 HTTP 请求处理中零散创建并发线程！
    - 当机械臂正在执行物理搬运任务 (is_busy) 时，自动暂停轮询，将底层 Socket 信道 100% 留给动作执行器，
      彻底消除 TCP 粘包与指令排队！
    - 在待机和拖拽示教时，平稳刷新机械臂末端坐标、关节角度与夹爪开度至 telemetry 内存。
    """
    while True:
        try:
            with state_lock:
                busy = telemetry.get("is_busy", False)
                connected = driver.is_connected

            if connected and not busy:
                c = driver.get_coords()
                a = driver.get_angles()
                g = driver.get_gripper_value()
                with state_lock:
                    if c and len(c) >= 6:
                        telemetry["coords"] = c
                    if a and len(a) >= 6:
                        telemetry["angles"] = a
                    if g is not None:
                        telemetry["gripper_val"] = g
        except Exception:
            pass
        time.sleep(0.5)

def refresh_pose(force: bool = False):
    """已重构为空操作，所有前端接口直接读取 telemetry 内存缓存，无延迟、无并发阻塞"""
    pass

_hud_font = None

def get_hud_font(font_size=15):
    global _hud_font
    if _hud_font is None:
        for fpath in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/arial.ttf']:
            if os.path.exists(fpath):
                try:
                    _hud_font = ImageFont.truetype(fpath, font_size)
                    break
                except Exception:
                    pass
        if _hud_font is None:
            _hud_font = ImageFont.load_default()
    return _hud_font

def qwen_live_perception_loop():
    """
    Qwen3-VL 8B 全局异步具身视觉感知守护线程：
    周期性（每 3.5 秒）调用云端 Qwen3-VL 目标检测模型，将工件检测框与映射物理坐标存入 telemetry。
    在机械臂运动中 (is_busy) 或示教中 (drag_teach) 自动暂停，释放云端 GPU 与 SSH 隧道带宽，
    确保用户发送动作/问答指令时能够秒级响应，零排队延迟。
    """
    time.sleep(1.5)
    while True:
        try:
            with state_lock:
                is_busy = telemetry.get("is_busy", False)
                drag_mode = telemetry.get("drag_teach", False)
            
            if not is_busy and not drag_mode and cam.is_running:
                frame = cam.get_latest_frame()
                if frame is not None:
                    detected_objs = detect_all_table_objects(frame, mapper=mapper)
                    if detected_objs is not None:
                        with state_lock:
                            telemetry["live_detected_objects"] = detected_objs
                            if detected_objs:
                                telemetry["cloud_status"] = "ONLINE"
            time.sleep(3.5)
        except Exception:
            time.sleep(3.5)

def generate_video_stream():
    """实时视频流生成器 (Qwen3-VL 实时动态感知 + HUD 准星与物理坐标投影，满帧 30 FPS 无阻塞)"""
    frame_idx = 0

    while cam.is_running:
        frame = cam.get_latest_frame()
        if frame is None:
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "CAMERA STANDBY / INITIALIZING...", (320, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 240, 255), 2)
            cv2.putText(placeholder, "FlexArm Embodied Cockpit Feed", (430, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 1)
            ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        disp = frame.copy()
        frame_idx += 1

        # 绘制手眼标定工作区边界轮廓或网格
        if hasattr(mapper, 'boundary_polygon') and mapper.boundary_polygon is not None and len(mapper.boundary_polygon) >= 3:
            b_pts = [[round(float(gx) * w / 1000.0), round(float(gy) * h / 1000.0)] for gx, gy in mapper.boundary_polygon]
            cv2.polylines(disp, [np.array(b_pts, np.int32)], True, (0, 240, 255), 2)
            cv2.putText(disp, "54P-Calibrated", (b_pts[0][0] + 5, max(20, b_pts[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
        elif mapper.grid_points is not None and (not mapper.image_size or list(mapper.image_size) == [w, h]):
            for gx, gy in mapper.grid_points:
                px, py = round(float(gx) * w / 1000.0), round(float(gy) * h / 1000.0)
                cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 9, 1)
            if len(mapper.image_points) == 9:
                for idx, (gx, gy) in enumerate(mapper.image_points):
                    px, py = round(float(gx) * w / 1000.0), round(float(gy) * h / 1000.0)
                    cv2.circle(disp, (px, py), 5, (0, 0, 255), -1)
                    cv2.circle(disp, (px, py), 10, (0, 255, 0), 1)
                    cv2.putText(disp, f"P{idx+1}", (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                dot_labels = ("点4(左上)", "点3(右上)", "点2(右下)", "点1(左下)")
                quad_pts = []
                for (gx, gy), label in zip(mapper.image_points, dot_labels):
                    px, py = round(float(gx) * w / 1000.0), round(float(gy) * h / 1000.0)
                    quad_pts.append([px, py])
                    cv2.circle(disp, (px, py), 6, (0, 0, 255), -1)
                    cv2.circle(disp, (px, py), 12, (0, 255, 255), 2)
                    cv2.putText(disp, label, (min(w - 70, px + 8), max(15, min(h - 5, py - 8))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if len(quad_pts) == 4:
                    cv2.polylines(disp, [np.array(quad_pts, np.int32)], True, (255, 120, 0), 1)

        # 象棋 9x10 物理棋盘实时透视网格渲染 (对弈投影)
        if telemetry.get("show_xiangqi_grid", True) or (xiangqi_agent and xiangqi_agent.is_running):
            try:
                # 绘制 10 条横线 (rank 0..9)
                for r_idx in range(10):
                    p_left = xiangqi_grid_mapper.get_square_pixel_pos(f"a{r_idx}", w, h)
                    p_right = xiangqi_grid_mapper.get_square_pixel_pos(f"i{r_idx}", w, h)
                    cv2.line(disp, p_left, p_right, (0, 165, 255), 1, cv2.LINE_AA)
                # 绘制 9 条竖线 (file a..i)
                for col_char in "abcdefghi":
                    p_bot = xiangqi_grid_mapper.get_square_pixel_pos(f"{col_char}0", w, h)
                    p_top = xiangqi_grid_mapper.get_square_pixel_pos(f"{col_char}9", w, h)
                    cv2.line(disp, p_bot, p_top, (0, 165, 255), 1, cv2.LINE_AA)
                # 绘制 90 个交叉点点位
                for r_idx in range(10):
                    for col_char in "abcdefghi":
                        pt = xiangqi_grid_mapper.get_square_pixel_pos(f"{col_char}{r_idx}", w, h)
                        cv2.circle(disp, pt, 3, (0, 255, 255), -1)
                # 绘制九宫格斜线 (红方: d0-f2, d2-f0; 黑方: d7-f9, d9-f7)
                cv2.line(disp, xiangqi_grid_mapper.get_square_pixel_pos("d0", w, h), xiangqi_grid_mapper.get_square_pixel_pos("f2", w, h), (0, 165, 255), 1, cv2.LINE_AA)
                cv2.line(disp, xiangqi_grid_mapper.get_square_pixel_pos("d2", w, h), xiangqi_grid_mapper.get_square_pixel_pos("f0", w, h), (0, 165, 255), 1, cv2.LINE_AA)
                cv2.line(disp, xiangqi_grid_mapper.get_square_pixel_pos("d7", w, h), xiangqi_grid_mapper.get_square_pixel_pos("f9", w, h), (0, 165, 255), 1, cv2.LINE_AA)
                cv2.line(disp, xiangqi_grid_mapper.get_square_pixel_pos("d9", w, h), xiangqi_grid_mapper.get_square_pixel_pos("f7", w, h), (0, 165, 255), 1, cv2.LINE_AA)
                # 标注 4 个角点标签
                for c_k, c_name in [("a0", "a0(红左)"), ("i0", "i0(红右)"), ("i9", "i9(黑右)"), ("a9", "a9(黑左)")]:
                    pt = xiangqi_grid_mapper.get_square_pixel_pos(c_k, w, h)
                    cv2.putText(disp, c_name, (pt[0] - 25, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # 动态走子箭头与高亮
                if xiangqi_agent.last_red_move:
                    m = xiangqi_agent.last_red_move
                    p1 = xiangqi_grid_mapper.get_square_pixel_pos(m[:2], w, h)
                    p2 = xiangqi_grid_mapper.get_square_pixel_pos(m[2:], w, h)
                    cv2.arrowedLine(disp, p1, p2, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
                    cv2.circle(disp, p2, 14, (0, 0, 255), 2)
                if xiangqi_agent.last_black_move:
                    m = xiangqi_agent.last_black_move
                    p1 = xiangqi_grid_mapper.get_square_pixel_pos(m[:2], w, h)
                    p2 = xiangqi_grid_mapper.get_square_pixel_pos(m[2:], w, h)
                    cv2.arrowedLine(disp, p1, p2, (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.25)
                    cv2.circle(disp, p2, 14, (255, 220, 0), 2)
            except Exception:
                pass

        # 从全局遥测数据中快速读取 Qwen3-VL 异步感知的工件列表（实时 30 FPS 高帧率推流，无任何网络延迟阻塞）
        with state_lock:
            cached_detected_objs = list(telemetry.get("live_detected_objects", []))
            off_x = telemetry.get("calib_offset_x", 0.0)
            off_y = telemetry.get("calib_offset_y", 0.0)
            active_bbox = telemetry.get("last_bbox")
            active_obj = telemetry.get("last_detected_obj")
            is_busy = telemetry.get("is_busy", False)

        labels_to_draw = []

        # 1. 实时动态渲染 Qwen3-VL 视野内的所有实体工件 (黄色急停盒、绿色胶带等)
        for obj in cached_detected_objs:
            ymin, xmin, ymax, xmax = obj["bbox"]
            px1, py1 = int(xmin * w / 1000.0), int(ymin * h / 1000.0)
            px2, py2 = int(xmax * w / 1000.0), int(ymax * h / 1000.0)
            color = obj.get("color_bgr", (0, 255, 120))
            name = obj.get("name", "工件")
            xyz = obj.get("robot_xyz")

            # 绘制实时动态外接矩形
            cv2.rectangle(disp, (px1, py1), (px2, py2), color, 2)
            
            # 四角科技感准星
            corner_len = min(15, max(5, (px2 - px1) // 5), max(5, (py2 - py1) // 5))
            cv2.line(disp, (px1, py1), (px1 + corner_len, py1), (255, 255, 255), 2)
            cv2.line(disp, (px1, py1), (px1, py1 + corner_len), (255, 255, 255), 2)
            cv2.line(disp, (px2, py1), (px2 - corner_len, py1), (255, 255, 255), 2)
            cv2.line(disp, (px2, py1), (px2, py1 + corner_len), (255, 255, 255), 2)
            cv2.line(disp, (px1, py2), (px1 + corner_len, py2), (255, 255, 255), 2)
            cv2.line(disp, (px1, py2), (px1, py2 - corner_len), (255, 255, 255), 2)
            cv2.line(disp, (px2, py2), (px2 - corner_len, py2), (255, 255, 255), 2)
            cv2.line(disp, (px2, py2), (px2, py2 - corner_len), (255, 255, 255), 2)

            # 中心红点与十字准星
            cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
            cv2.circle(disp, (cx, cy), 4, (0, 0, 255), -1)
            cv2.drawMarker(disp, (cx, cy), (0, 240, 255), cv2.MARKER_CROSS, 12, 1)

            # 动态标签与实时物理坐标投影 (含手眼微调偏置)
            label = f"{name}"
            if xyz:
                applied_x = round(xyz[0] + off_x, 1)
                applied_y = round(xyz[1] + off_y, 1)
                label += f" [{applied_x}, {applied_y}]"
            
            labels_to_draw.append(((px1, max(4, py1 - 22)), label, color))

        # 2. 若当前有任务正在执行 (is_busy)，在被锁定抓取的目标上绘制高亮提示框
        if is_busy and active_bbox:
            ymin, xmin, ymax, xmax = active_bbox
            px1, py1 = int(xmin * w / 1000.0), int(ymin * h / 1000.0)
            px2, py2 = int(xmax * w / 1000.0), int(ymax * h / 1000.0)
            cv2.rectangle(disp, (px1 - 4, py1 - 4), (px2 + 4, py2 + 4), (0, 165, 255), 2)
            labels_to_draw.append(((px1, max(24, py1 - 26)), f"锁定: {active_obj}", (0, 165, 255)))

        # 3. 统一使用微软雅黑高清平滑字体渲染中文与坐标标签 (单次快速图层融合)
        if labels_to_draw:
            img_pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = get_hud_font(15)
            for (lx, ly), text, color_bgr in labels_to_draw:
                tb = draw.textbbox((lx, ly), text, font=font)
                draw.rectangle((tb[0]-4, tb[1]-2, tb[2]+4, tb[3]+2), fill=(15, 23, 42))
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                draw.text((lx, ly), text, font=font, fill=color_rgb)
            disp = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 象棋模式视角聚焦：自动精准裁切棋盘区域并放大呈现
        if telemetry.get("focus_xiangqi_board", False) or (xiangqi_agent and xiangqi_agent.is_running):
            try:
                bx1, by1, bx2, by2 = xiangqi_grid_mapper.get_board_bounding_rect_px(w, h)
                pad_x = int((bx2 - bx1) * 0.08)
                pad_y = int((by2 - by1) * 0.08)
                fx1 = max(0, bx1 - pad_x)
                fy1 = max(0, by1 - pad_y)
                fx2 = min(w, bx2 + pad_x)
                fy2 = min(h, by2 + pad_y)
                if fx2 > fx1 + 50 and fy2 > fy1 + 50:
                    disp = disp[fy1:fy2, fx1:fx2]
            except Exception:
                pass

        # 编码为 JPEG
        ret, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)

# ==============================================================================
# 前端 HTML / CSS / JS 模板 (赛博工业风多功能具身操控台)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlexArm Embodied Cockpit | 机械臂具身视觉操控大屏</title>
    <style>
        :root {
            --bg-base: #0a0e17;
            --panel-bg: rgba(18, 24, 38, 0.92);
            --border-glow: #00f0ff;
            --accent-green: #00ff88;
            --accent-yellow: #f59e0b;
            --accent-red: #ff3366;
            --text-main: #e2e8f0;
            --text-muted: #8892b0;
            --btn-blue: #0284c7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background-color: var(--bg-base);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        header {
            background: #101623;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand h1 { font-size: 1.15rem; font-weight: 700; color: #00f0ff; letter-spacing: 1px; }
        .badge {
            background: rgba(0, 240, 255, 0.12);
            color: #00f0ff;
            border: 1px solid #00f0ff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
        }
        .main-container {
            display: grid;
            grid-template-columns: 1fr 480px;
            gap: 14px;
            padding: 14px;
            flex: 1;
            height: calc(100vh - 54px);
        }
        .panel {
            background: var(--panel-bg);
            border: 1px solid #1e293b;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .panel-header {
            background: #141c2e;
            padding: 9px 14px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
        }
        .video-wrapper {
            position: relative;
            flex: 1;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        #videoStream {
            max-width: 100%;
            max-height: 100%;
            cursor: crosshair;
        }
        .video-overlay-hint {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.75);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            color: #38bdf8;
            pointer-events: none;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .tabs {
            display: flex;
            background: #0d131f;
            border-bottom: 1px solid #1e293b;
        }
        .tab-btn {
            flex: 1;
            padding: 9px 12px;
            background: transparent;
            color: #94a3b8;
            border: none;
            cursor: pointer;
            font-size: 0.78rem;
            font-weight: 600;
            transition: all 0.2s;
            border-bottom: 2px solid transparent;
        }
        .tab-btn:hover { color: #fff; background: rgba(255,255,255,0.02); }
        .tab-btn.active {
            color: #00f0ff;
            background: rgba(0, 240, 255, 0.08);
            border-bottom: 2px solid #00f0ff;
        }
        .tab-content {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: none;
            flex-direction: column;
            gap: 12px;
        }
        .tab-content.active { display: flex; }
        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .stat-card {
            background: rgba(15, 23, 42, 0.6);
            padding: 8px 10px;
            border-radius: 6px;
            border: 1px solid #273549;
        }
        .stat-label { font-size: 0.68rem; color: var(--text-muted); }
        .stat-val { font-size: 0.88rem; font-weight: 700; color: #38bdf8; margin-top: 2px; }
        .input-group { display: flex; gap: 8px; }
        input[type="text"] {
            flex: 1;
            background: #0f172a;
            border: 1px solid #334155;
            color: #fff;
            padding: 9px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            outline: none;
        }
        input[type="text"]:focus { border-color: var(--border-glow); }
        button {
            background: var(--btn-blue);
            color: white;
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.8rem;
            transition: all 0.2s;
        }
        button:hover { background: #0369a1; }
        .btn-voice { background: #475569; width: 40px; display: flex; align-items: center; justify-content: center; }
        .btn-voice.listening { background: var(--accent-red); animation: pulse 1s infinite; }
        .action-pills { display: flex; flex-wrap: wrap; gap: 6px; }
        .pill {
            background: #1e293b;
            border: 1px solid #334155;
            color: #94a3b8;
            padding: 5px 9px;
            border-radius: 4px;
            font-size: 0.72rem;
            cursor: pointer;
        }
        .pill:hover { background: #334155; color: #fff; }
        .btn-danger { background: var(--accent-red); width: 100%; }
        .btn-danger:hover { background: #dc2626; }
        .btn-warning { background: #d97706; }
        .btn-warning:hover { background: #b45309; }
        .btn-success { background: #059669; }
        .btn-success:hover { background: #047857; }
        .progress-bar {
            height: 6px;
            background: #1e293b;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-fill { height: 100%; width: 0%; background: var(--accent-green); transition: width 0.3s; }
        .log-box {
            background: #060913;
            padding: 8px 10px;
            border-radius: 6px;
            border: 1px solid #1e293b;
            font-family: monospace;
            font-size: 0.72rem;
            color: #10b981;
            height: 150px;
            overflow-y: auto;
        }
        /* 点动控制盘样式 */
        .jog-section {
            background: rgba(15, 23, 42, 0.5);
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #273549;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .jog-title { font-size: 0.75rem; color: #38bdf8; font-weight: 600; display: flex; justify-content: space-between; }
        .jog-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
        }
        .btn-jog {
            background: #1e293b;
            border: 1px solid #334155;
            padding: 8px 4px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .btn-jog:hover { background: #0284c7; border-color: #38bdf8; }
        .step-selector { display: flex; gap: 6px; align-items: center; font-size: 0.72rem; color: #94a3b8; }
        .step-badge {
            background: #1e293b;
            padding: 2px 8px;
            border-radius: 3px;
            cursor: pointer;
            border: 1px solid #334155;
        }
        .step-badge.active { background: #0284c7; color: white; border-color: #38bdf8; }
        /* 夹爪控制 */
        .slider-row { display: flex; align-items: center; gap: 10px; }
        input[type="range"] { flex: 1; accent-color: #00f0ff; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>FLEXARM COCKPIT</h1>
            <span class="badge">EMBODIED VLA & VLM</span>
            <span class="badge" style="border-color:#10b981; color:#10b981;">HARDWARE ACCELERATED</span>
        </div>
        <div id="connectionStatus" style="font-size:0.78rem; color:#10b981;">● CLOUD ONLINE (1.9s)</div>
    </header>

    <div class="main-container">
        <!-- 视频监控工作区 -->
        <div class="panel">
            <div class="panel-header">
                <span>🔴 工作台实时视频流 (REALTIME FEED & HUD)</span>
                <select id="clickMode" aria-label="画面点击模式" style="background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:4px; padding:4px; font-size:0.75rem;">
                    <option value="preview">定位预览（不运动）</option>
                    <option value="pick" selected>抓取并送往 P5（使用当前抓取高度）</option>
                </select>
            </div>
            <div class="video-wrapper">
                <img id="videoStream" src="/video_feed" alt="Camera Feed" onclick="handleCanvasClick(event)" onmousemove="handleCanvasMouseMove(event)">
                <div class="video-overlay-hint">💡 点击后确认抓取；可切换定位预览，绿色网格用于核对位置</div>
            </div>
            <div style="padding: 8px 14px; background: #0e1524; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1e293b;">
                <div style="font-size: 0.75rem; color: #94a3b8;">
                    末端位姿: <span id="hudCoords" style="color:#38bdf8; font-weight:700;">X:-- Y:-- Z:--</span>
                    | 夹爪开度: <span id="hudGripper" style="color:#00ff88; font-weight:700;">15%</span>
                    | <span id="hoverCoords" style="color:#fde047; font-family: monospace;">光标: [---, ---]</span>
                    | <span id="positionPreview" style="color:#a7f3d0;">切换定位预览可查看 X、Y</span>
                </div>
                <button class="btn-danger" style="width: auto; padding: 4px 12px; font-size:0.75rem;" onclick="sendManual('emergency_stop')">🛑 紧急制动 (E-STOP)</button>
            </div>
            <!-- 手眼偏置微调补偿控制条 -->
            <div style="padding: 8px 14px; background: #0b1120; border-top: 1px solid #1e293b; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">🎯 手眼微调偏置 (mm):</span>
                <span id="offsetSummary" style="font-size: 0.75rem; color: #a7f3d0; font-family: monospace; font-weight: bold;">ΔX:+0.0 ΔY:+0.0 ΔZ:+0.0</span>
                <div style="display:flex; gap:8px; margin-left:auto; flex-wrap:wrap; align-items:center;">
                    <div style="display:flex; gap:4px;">
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('x', 5.0)">X+5</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('x', -5.0)">X-5</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('x', 1.0)">X+1</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('x', -1.0)">X-1</button>
                    </div>
                    <div style="display:flex; gap:4px;">
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('y', 5.0)">Y+5</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('y', -5.0)">Y-5</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('y', 1.0)">Y+1</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('y', -1.0)">Y-1</button>
                    </div>
                    <div style="display:flex; gap:4px;">
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('z', 5.0)">Z+5</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('z', -5.0)">Z-5</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('z', 1.0)">Z+1</button>
                        <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem;" onclick="nudgeCalibOffset('z', -1.0)">Z-1</button>
                    </div>
                    <button class="btn-jog" style="padding:3px 8px; font-size:0.72rem; background:#334155; color:#f87171;" onclick="resetCalibOffset()">偏置归零</button>
                </div>
            </div>
        </div>

        <!-- 右侧仪表与控制面板 -->
        <div class="panel">
            <!-- 标签页导航 -->
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('tab-vlm', this)">🧠 视觉大模型</button>
                <button class="tab-btn" onclick="switchTab('tab-jog', this)">🎮 全向点动位姿</button>
                <button class="tab-btn" onclick="switchTab('tab-tools', this)">🛠️ 具身工具箱</button>
                <button class="tab-btn" onclick="switchTab('tab-xiangqi', this)">♟️ 人机对弈 (Pikafish)</button>
            </div>

            <!-- Tab 1: 视觉智能控制 (VLM AI) -->
            <div class="tab-content active" id="tab-vlm">
                <!-- 遥测状态卡片 -->
                <div class="telemetry-grid">
                    <div class="stat-card">
                        <div class="stat-label">机械臂连接状态</div>
                        <div class="stat-val" id="robotStatus">...</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">视觉大模型耗时</div>
                        <div class="stat-val" id="latencyVal">1941 ms</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">最近识别目标</div>
                        <div class="stat-val" id="lastObj">无</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">物理映射坐标</div>
                        <div class="stat-val" id="lastXYZ">X:-- Y:--</div>
                    </div>
                </div>

                <!-- 动作工序进度条 -->
                <div>
                    <div style="font-size:0.72rem; padding-bottom: 4px; color:#94a3b8; display:flex; justify-content:space-between;">
                        <span id="stepText">ACT-00 待机中</span>
                        <span id="stepPercent">0%</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                </div>

                <!-- 自然语言交互 -->
                <div class="input-group">
                    <input type="text" id="cmdInput" placeholder="输入指令，如: 把工件移到P4弃子区..." onkeydown="if(event.key==='Enter') sendCommand()">
                    <button class="btn-voice" id="voiceBtn" onclick="toggleVoice()" title="语音输入">🎤</button>
                    <button onclick="sendCommand()">发送</button>
                </div>

                <!-- 快捷工序指令 -->
                <div style="font-size:0.72rem; color:#64748b;">快捷预设工序:</div>
                <div class="action-pills">
                    <span class="pill" style="border-color:#f59e0b; color:#fbbf24;" onclick="sendQuick('把工作台上的工件抓取到P4弃子区')">📦 移出工件->P4</span>
                    <span class="pill" style="border-color:#10b981; color:#34d399;" onclick="sendQuick('把桌上的胶带抓取到P5工位')">📦 抓取胶带->P5</span>
                    <span class="pill" onclick="sendQuick('扫视一下工作台，你会发现一个胶带，给他移到p5')">🎯 扫视台面胶带->移到P5</span>
                    <span class="pill" onclick="sendQuick('扫描工作台上的所有零件')">🔍 全景巡检扫描</span>
                    <span class="pill" onclick="sendQuick('机械臂复位到待机状态')">🏠 安全复位(NORMAL)</span>
                    <span class="pill" onclick="sendQuick('机械臂挥手打招呼')">👋 挥手问好</span>
                </div>

                <!-- 实时日志 -->
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">📋 运行动态日志</div>
                <div class="log-box" id="logBox"></div>
            </div>

            <!-- Tab 2: 机械臂全向微调点动与位姿 (Jog & Pose) -->
            <div class="tab-content" id="tab-jog">
                <!-- 步长选择 -->
                <div class="step-selector">
                    <span>微调步长:</span>
                    <span class="step-badge" onclick="setJogStep(1, this)">1mm / 1°</span>
                    <span class="step-badge active" onclick="setJogStep(5, this)">5mm / 5°</span>
                    <span class="step-badge" onclick="setJogStep(10, this)">10mm / 10°</span>
                    <span class="step-badge" onclick="setJogStep(25, this)">25mm / 20°</span>
                </div>

                <!-- 笛卡尔坐标点动控制盘 (XYZ) -->
                <div class="jog-section">
                    <div class="jog-title">
                        <span>📐 笛卡尔空间点动 (CARTESIAN JOG)</span>
                        <span id="posDisplay" style="color:#94a3b8; font-weight:normal;">X:180 Y:0 Z:180</span>
                    </div>
                    <div class="jog-grid-3">
                        <button class="btn-jog" onclick="jogCoord('x', 1)">+X (前伸)</button>
                        <button class="btn-jog" onclick="jogCoord('y', 1)">+Y (左移)</button>
                        <button class="btn-jog" onclick="jogCoord('z', 1)">+Z (抬升)</button>
                        <button class="btn-jog" onclick="jogCoord('x', -1)">-X (后缩)</button>
                        <button class="btn-jog" onclick="jogCoord('y', -1)">-Y (右移)</button>
                        <button class="btn-jog" onclick="jogCoord('z', -1)">-Z (下压)</button>
                    </div>
                    <!-- 姿态旋转角 Rx Ry Rz -->
                    <div class="jog-grid-3" style="margin-top: 4px;">
                        <button class="btn-jog" onclick="jogCoord('rx', 1)">+Rx 俯仰</button>
                        <button class="btn-jog" onclick="jogCoord('ry', 1)">+Ry 翻滚</button>
                        <button class="btn-jog" onclick="jogCoord('rz', 1)">+Rz 偏航</button>
                        <button class="btn-jog" onclick="jogCoord('rx', -1)">-Rx 俯仰</button>
                        <button class="btn-jog" onclick="jogCoord('ry', -1)">-Ry 翻滚</button>
                        <button class="btn-jog" onclick="jogCoord('rz', -1)">-Rz 偏航</button>
                    </div>
                </div>

                <!-- 6 轴关节独立调谐 (Joints) -->
                <div class="jog-section">
                    <div class="jog-title">
                        <span>⚙️ 关节独立调谐 (JOINT ADJUST)</span>
                        <span id="jointDisplay" style="color:#94a3b8; font-weight:normal;">J1~J6</span>
                    </div>
                    <div class="jog-grid-3">
                        <button class="btn-jog" onclick="jogJoint(1, 1)">J1 (+)</button>
                        <button class="btn-jog" onclick="jogJoint(2, 1)">J2 (+)</button>
                        <button class="btn-jog" onclick="jogJoint(3, 1)">J3 (+)</button>
                        <button class="btn-jog" onclick="jogJoint(1, -1)">J1 (-)</button>
                        <button class="btn-jog" onclick="jogJoint(2, -1)">J2 (-)</button>
                        <button class="btn-jog" onclick="jogJoint(3, -1)">J3 (-)</button>
                    </div>
                    <div class="jog-grid-3" style="margin-top: 4px;">
                        <button class="btn-jog" onclick="jogJoint(4, 1)">J4 (+)</button>
                        <button class="btn-jog" onclick="jogJoint(5, 1)">J5 (+)</button>
                        <button class="btn-jog" onclick="jogJoint(6, 1)">J6 (+)</button>
                        <button class="btn-jog" onclick="jogJoint(4, -1)">J4 (-)</button>
                        <button class="btn-jog" onclick="jogJoint(5, -1)">J5 (-)</button>
                        <button class="btn-jog" onclick="jogJoint(6, -1)">J6 (-)</button>
                    </div>
                </div>

                <!-- 自适应夹爪连续滑块 -->
                <div class="jog-section">
                    <div class="jog-title">
                        <span>🤏 夹爪行程精控 (GRIPPER)</span>
                        <span id="gripperSliderVal">15% (开)</span>
                    </div>
                    <div class="slider-row">
                        <span style="font-size:0.72rem; color:#94a3b8;">闭合</span>
                        <input type="range" id="gripperRange" min="0" max="100" value="15" oninput="onGripperSliderChange(this.value)">
                        <span style="font-size:0.72rem; color:#94a3b8;">张开</span>
                    </div>
                    <div style="display:flex; gap:6px; margin-top:4px;">
                        <button style="flex:1;" class="btn-jog" onclick="setGripperPreset(15)">开爪 (15%)</button>
                        <button style="flex:1;" class="btn-jog" onclick="setGripperPreset(10)">半开 (10%)</button>
                        <button style="flex:1;" class="btn-jog" onclick="setGripperPreset(5)">收爪 (5%)</button>
                    </div>
                    <button style="width:100%; margin-top:6px; font-size:0.75rem; background:#1e293b; color:#38bdf8; border:1px solid #0284c7; padding:5px; border-radius:4px; cursor:pointer;" onclick="initGripperHardware()">🔄 485 Modbus 夹爪自检重连 (ID: 14)</button>
                </div>

                <!-- 机械臂运行速度调节 -->
                <div class="jog-section">
                    <div class="jog-title">
                        <span>⚡ 移动速度精控 (SPEED)</span>
                        <span id="speedSliderVal" style="color:#00f0ff; font-weight:bold;">20% (标准/稳健)</span>
                    </div>
                    <div class="slider-row">
                        <span style="font-size:0.72rem; color:#94a3b8;">5%</span>
                        <input type="range" id="speedRange" min="5" max="80" value="20" step="5" oninput="onSpeedSliderChange(this.value)">
                        <span style="font-size:0.72rem; color:#94a3b8;">80%</span>
                    </div>
                    <div style="display:flex; gap:6px; margin-top:4px;">
                        <button style="flex:1;" class="btn-jog" onclick="setSpeedPreset(10, '慢速/精细')">微速 10%</button>
                        <button style="flex:1;" class="btn-jog" onclick="setSpeedPreset(20, '标准/稳健')">标准 20%</button>
                        <button style="flex:1;" class="btn-jog" onclick="setSpeedPreset(35, '快速/敏捷')">快速 35%</button>
                        <button style="flex:1;" class="btn-jog" onclick="setSpeedPreset(50, '疾速/高效')">疾速 50%</button>
                    </div>
                </div>

                <!-- 标定工位一键巡航 -->
                <div class="jog-section">
                    <div class="jog-title"><span>📍 预设工位直达 (STATIONS)</span></div>
                    <div class="jog-grid-3">
                        <button class="btn-jog" style="background:#0284c7; color:#fff;" onclick="goToStation('NORMAL')">🏠 NORMAL (待机)</button>
                        <button class="btn-jog" onclick="goToStation('P1')">📦 P1 工位</button>
                        <button class="btn-jog" onclick="goToStation('P2')">📦 P2 工位</button>
                        <button class="btn-jog" onclick="goToStation('P4')">📦 P4 (弃子/暂存)</button>
                        <button class="btn-jog" onclick="goToStation('P5')">📦 P5 工位</button>
                    </div>
                </div>
            </div>

            <!-- Tab 3: 具身工具中心与高级功能 (Tool Center & OpenVLA) -->
            <div class="tab-content" id="tab-tools">
                <!-- 自由拖拽示教与点位录制 -->
                <div class="jog-section">
                    <div class="jog-title">
                        <span>🖐️ 自由拖动示教与点位录制 (DRAG TEACH)</span>
                        <span id="teachStatusBadge" style="color:#00ff88; font-size:0.72rem;">状态: 刚性保持</span>
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8;">降低阻尼后可直接用手移动机械臂，移动至目标姿态点击【记录】即可加入轨迹序列。</div>
                    <div style="display:flex; gap:8px; margin-top:6px;">
                        <button id="teachBtn" class="btn-warning" style="flex:1;" onclick="toggleDragTeach(true)">🔓 降低阻尼 (自由拖动)</button>
                        <button class="btn-success" style="flex:1;" onclick="toggleDragTeach(false)">🔒 锁定舵机 (恢复刚性)</button>
                    </div>
                    <div style="display:flex; gap:6px; margin-top:6px;">
                        <button class="btn-jog" style="flex:2; background:#0284c7; color:#fff;" onclick="recordWaypointUI()">📍 记录当前状态为点位</button>
                        <button class="btn-jog" style="flex:1; background:#059669; color:#fff;" onclick="replayWaypointsUI()">▶️ 轨迹回放</button>
                        <button class="btn-jog" style="flex:1; background:#dc2626; color:#fff;" onclick="clearWaypointsUI()">🗑️ 清空</button>
                    </div>
                    <div id="waypointTableBox" style="max-height:120px; overflow-y:auto; background:#070b14; border:1px solid #1e293b; border-radius:4px; padding:6px; font-size:0.7rem; font-family:monospace; color:#38bdf8; margin-top:6px;">
                        暂未记录点位 (拖动机械臂后点击上方按钮即可记录)
                    </div>
                </div>

                <!-- 头部状态指示灯语 -->
                <div class="jog-section">
                    <div class="jog-title"><span>💡 头部 RGB LED 灯语切换</span></div>
                    <div class="jog-grid-3">
                        <button class="btn-jog" style="border-left: 3px solid #00ff88;" onclick="setLed('IDLE')">🟢 正常待机</button>
                        <button class="btn-jog" style="border-left: 3px solid #38bdf8;" onclick="setLed('VISION')">🔵 视觉思考</button>
                        <button class="btn-jog" style="border-left: 3px solid #f59e0b;" onclick="setLed('MOVING')">🟡 运动执行</button>
                        <button class="btn-jog" style="border-left: 3px solid #ff3366;" onclick="setLed('ERROR')">🔴 异常报警</button>
                        <button class="btn-jog" style="border-left: 3px solid #ffffff;" onclick="setLed('LIGHT')">⚪ 视觉补光</button>
                        <button class="btn-jog" style="border-left: 3px solid #64748b;" onclick="setLed('OFF')">⚫ 关闭灯光</button>
                    </div>
                </div>

                <!-- 全景巡检扫描 -->
                <div class="jog-section">
                    <div class="jog-title"><span>🔍 多视点全景扫描 (PANORAMIC SCAN)</span></div>
                    <div style="font-size:0.72rem; color:#94a3b8;">机械臂按左-中-右三维航线巡视工作台，全面识别遗漏物料。</div>
                    <button class="btn-jog" style="margin-top:6px; background:#0369a1;" onclick="startScan()">🚀 启动三维全景扫描巡检</button>
                </div>

                <!-- Qwen3.8-Flash 具身规划与单步动作模拟测试 -->
                <div class="jog-section">
                    <div class="jog-title"><span>🤖 Qwen3.8-Flash 具身动作推演</span></div>
                    <div style="font-size:0.72rem; color:#94a3b8;">基于 Qwen3.8-Flash (Vision) 多模态视觉大模型推理并物理执行:</div>
                    <div style="display:flex; gap:6px; margin-top:6px;">
                        <button class="btn-jog" onclick="testOpenVLAStep([0.5, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0])">下探抓取步进</button>
                        <button class="btn-jog" onclick="testOpenVLAStep([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])">抬升释放步进</button>
                    </div>
                </div>
            </div>

            <!-- Tab 4: 具身智能象棋对弈 (Xiangqi AI + Pikafish) -->
            <div class="tab-content" id="tab-xiangqi">
                <div class="telemetry-grid">
                    <div class="stat-card">
                        <div class="stat-label">对弈流水线状态</div>
                        <div class="stat-val" id="xqState" style="color:#38bdf8; font-size:0.95rem;">IDLE (待机)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">当前执子方 (TURN)</div>
                        <div class="stat-val" id="xqTurn" style="font-size:0.95rem;">红方 (人类玩家)</div>
                    </div>
                </div>

                <!-- 流程状态卡片 -->
                <div class="jog-section" style="border-left: 3px solid #38bdf8;">
                    <div class="jog-title"><span>🤖 具身 5 步智能对弈闭环流转</span></div>
                    <div id="xqStatusText" style="font-size:0.8rem; color:#e2e8f0; margin-top:4px; line-height:1.4;">
                        尚未启动对弈。点击下方【🚀 启动人机对弈闭环】即刻开始！
                    </div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:6px;">
                        流程：[1.拍基准图] ➔ [2.OpenCV静止检测人手撤回] ➔ [3.网格差分与规则校验] ➔ [4.Pikafish毫秒决策] ➔ [5.机械臂自动下子/吃子] ➔ 循环
                    </div>
                    <div style="display:flex; gap:8px; margin-top:10px;">
                        <button id="btnXqStart" class="btn-success" style="flex:2;" onclick="startXqGame()">🚀 启动对弈闭环</button>
                        <button id="btnXqEndgame" class="btn-primary" style="flex:2;" onclick="scanAndStartEndgame()">🧩 残局视觉识别与启动</button>
                        <button id="btnXqStop" class="btn-danger" style="flex:2; display:none;" onclick="stopXqGame()">⏸️ 暂停对弈 (Pause)</button>
                        <button class="btn-warning" style="flex:1;" onclick="resetXqGame()">🔄 重置</button>
                    </div>
                    <div id="xqEndgameStatus" style="font-size:0.75rem; color:#fde047; margin-top:6px; display:none;"></div>
                </div>

                <!-- FEN 与上步走子 -->
                <div class="jog-section">
                    <div class="jog-title"><span>📋 实时棋盘状态 (FEN & MOVES)</span></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-top:4px;">
                        <span style="color:#94a3b8;">上步着法:</span>
                        <span id="xqLastMove" style="color:#fde047; font-family:monospace; font-weight:bold;">红: -- | 黑: --</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-top:4px;">
                        <span style="color:#94a3b8;">手眼微调补偿:</span>
                        <span id="xqOffsetSummary" style="color:#38bdf8; font-family:monospace; font-weight:bold;">ΔX:+0.0 ΔY:+0.0 ΔZ:+0.0 mm</span>
                    </div>
                    <div style="font-size:0.7rem; color:#64748b; margin-top:4px; word-break:break-all; font-family:monospace; background:#0f172a; padding:4px 6px; border-radius:4px;" id="xqFen">
                        rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
                    </div>
                </div>

                <!-- 历史走法记录列表 -->
                <div class="jog-section">
                    <div class="jog-title"><span>📜 对局着法记录 (MOVE HISTORY)</span></div>
                    <div id="xqHistory" style="max-height:130px; overflow-y:auto; margin-top:4px; background:#0f172a; padding:6px; border-radius:4px;">
                        <div style="color:#64748b; font-size:0.75rem;">暂无走子记录，点击上方启动按钮开始对弈</div>
                    </div>
                </div>

                <!-- 棋盘标定与视觉网格辅助 -->
                <div class="jog-section">
                    <div class="jog-title"><span>📐 棋盘 4 角点视觉对齐与自动定位</span></div>
                    <div style="font-size:0.72rem; color:#94a3b8;">画面实时投影 9x10 象棋网格。如果移动了棋盘位置，可点击一键自动定位重新标定：</div>
                    <div style="display:flex; gap:6px; margin-top:8px;">
                        <button id="btnXqAutoLocate" class="btn-primary" style="flex:2;" onclick="autoLocateXqBoard()">🎯 一键自动定位棋盘</button>
                        <button id="btnXqFocus" class="btn-jog" style="flex:1;" onclick="toggleXqFocus()">🔍 聚焦棋盘</button>
                        <button class="btn-jog" style="flex:1;" onclick="toggleXqGrid()">👁️ 切换网格</button>
                    </div>
                    <div id="xqLocateStatus" style="font-size:0.72rem; color:#38bdf8; margin-top:5px; display:none;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentJogStep = 5;

        function switchTab(tabId, el) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            const btn = el || (window.event ? (window.event.currentTarget || window.event.target) : null);
            if (btn && btn.classList) btn.classList.add('active');
            const targetContent = document.getElementById(tabId);
            if (targetContent) targetContent.classList.add('active');
        }

        function setJogStep(val, el) {
            currentJogStep = val;
            document.querySelectorAll('.step-badge').forEach(b => b.classList.remove('active'));
            el.classList.add('active');
        }

        // 自动轮询遥测状态
        function updateTelemetry() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('robotStatus').innerText = data.robot_status;
                    document.getElementById('latencyVal').innerText = data.last_latency_ms.toFixed(0) + ' ms';
                    document.getElementById('lastObj').innerText = data.last_detected_obj;
                    document.getElementById('lastXYZ').innerText = data.last_coords ? `X:${data.last_coords[0]} Y:${data.last_coords[1]}` : '--';
                    document.getElementById('stepText').innerText = data.current_step;
                    document.getElementById('stepPercent').innerText = data.step_progress + '%';
                    document.getElementById('progressFill').style.width = data.step_progress + '%';

                    const connBadge = document.getElementById('connectionStatus');
                    if (connBadge) {
                        if (data.cloud_status === 'ONLINE') {
                            const modelLabel = data.cloud_model || 'Qwen3.8-Flash';
                            connBadge.innerHTML = `<span style="color:#10b981;">● CLOUD ONLINE (${modelLabel})</span>`;
                        } else {
                            connBadge.innerHTML = '<span style="color:#f59e0b;">● EDGE ACTIVE (本地具身智能中枢)</span>';
                        }
                    }

                    if (data.coords) {
                        const c = data.coords;
                        const txt = `X:${c[0].toFixed(0)} Y:${c[1].toFixed(0)} Z:${c[2].toFixed(0)}`;
                        document.getElementById('hudCoords').innerText = txt;
                        document.getElementById('posDisplay').innerText = txt;
                    }
                    if (data.angles) {
                        const a = data.angles;
                        document.getElementById('jointDisplay').innerText = `[${a.map(v => v.toFixed(0)).join(', ')}]`;
                    }
                    if (data.gripper_val !== undefined) {
                        document.getElementById('hudGripper').innerText = data.gripper_val + '%';
                        document.getElementById('gripperSliderVal').innerText = data.gripper_val + '%';
                        document.getElementById('gripperRange').value = data.gripper_val;
                    }
                    if (data.calib_offset_x !== undefined) {
                        const ox = (data.calib_offset_x >= 0 ? '+' : '') + Number(data.calib_offset_x).toFixed(1);
                        const oy = (data.calib_offset_y >= 0 ? '+' : '') + Number(data.calib_offset_y).toFixed(1);
                        const oz = (data.calib_offset_z >= 0 ? '+' : '') + Number(data.calib_offset_z).toFixed(1);
                        const summary = `ΔX:${ox} ΔY:${oy} ΔZ:${oz}`;
                        const el = document.getElementById('offsetSummary');
                        if (el) el.innerText = summary;
                    }

                    // 刷新日志
                    const logBox = document.getElementById('logBox');
                    logBox.innerHTML = data.logs.join('<br>');
                    logBox.scrollTop = logBox.scrollHeight;
                });
        }
        setInterval(updateTelemetry, 1000);

        function showControlResult(response) {
            return response.json().then(data => {
                if (!data.success) alert(data.error || '操作未执行，请检查机械臂状态');
                updateTelemetry();
                return data;
            }).catch(() => alert('无法读取控制结果，请检查控制服务'));
        }

        // 笛卡尔空间单轴点动
        function jogCoord(axis, sign) {
            const delta = sign * currentJogStep;
            fetch('/api/jog_cartesian', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({axis: axis, delta: delta})
            }).then(showControlResult);
        }

        // 关节角度微调
        function jogJoint(jointIdx, sign) {
            const delta = sign * currentJogStep;
            fetch('/api/jog_joint', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({joint_idx: jointIdx, delta_deg: delta})
            }).then(showControlResult);
        }

        // 夹爪连续滑块
        function onGripperSliderChange(val) {
            document.getElementById('gripperSliderVal').innerText = val + '%';
            fetch('/api/gripper_slider', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({value: parseInt(val)})
            }).then(r => r.json()).then(data => {
                if (!data.success) document.getElementById('positionPreview').innerText = data.error || '夹爪操作未执行';
            });
        }

        function setGripperPreset(val) {
            document.getElementById('gripperRange').value = val;
            onGripperSliderChange(val);
        }

        // 运行速度连续调节
        function onSpeedSliderChange(val) {
            let tag = val <= 12 ? '慢速/精细' : (val <= 25 ? '标准/稳健' : (val <= 45 ? '快速/敏捷' : '疾速/高效'));
            document.getElementById('speedSliderVal').innerText = val + '% (' + tag + ')';
            fetch('/api/set_speed', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({speed: parseInt(val)})
            });
        }

        function setSpeedPreset(val, tag) {
            document.getElementById('speedRange').value = val;
            document.getElementById('speedSliderVal').innerText = val + '% (' + tag + ')';
            fetch('/api/set_speed', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({speed: parseInt(val)})
            });
        }

        function initGripperHardware() {
            fetch('/api/init_gripper', {method: 'POST'})
                .then(r => r.json())
                .then(res => {
                    alert(res.success ? `✅ Pro 夹爪初始化成功！实测开度: ${res.gripper_val}%` : (res.error || '夹爪初始化未完成'));
                    updateTelemetry();
                });
        }

        // 巡航至工位
        function goToStation(name) {
            fetch('/api/move_station', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({station: name})
            }).then(showControlResult);
        }

        // 示教模式
        function toggleDragTeach(enable) {
            fetch('/api/drag_teach', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enable: enable})
            }).then(r => r.json()).then(data => {
                if (!data.success) {
                    alert(data.error || '示教模式切换失败');
                    return;
                }
                updateTelemetry();
                const badge = document.getElementById('teachStatusBadge');
                if (enable) {
                    badge.innerText = '状态: 自由拖动中 (阻尼已降)';
                    badge.style.color = '#f59e0b';
                } else {
                    badge.innerText = '状态: 刚性保持';
                    badge.style.color = '#00ff88';
                }
            });
        }

        // 点位录制与回放 UI
        function recordWaypointUI() {
            const name = prompt('请输入点位标签名称 (直接点击确定将默认命名为 P1, P2...):', '');
            if (name === null) return;
            fetch('/api/record_waypoint', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name.trim()})
            }).then(r => r.json()).then(data => {
                if (!data.success) {
                    alert(data.error || '点位记录失败');
                    return;
                }
                refreshWaypointsList();
                updateTelemetry();
            });
        }

        function replayWaypointsUI() {
            if (confirm('确认要在真机上慢速回放当前已录制的所有点位吗？')) {
                fetch('/api/replay_waypoints', {method: 'POST'}).then(showControlResult);
            }
        }

        function clearWaypointsUI() {
            if (confirm('确认清空当前已录制的轨迹点位吗？')) {
                fetch('/api/clear_waypoints', {method: 'POST'}).then(() => {
                    refreshWaypointsList();
                    updateTelemetry();
                });
            }
        }

        function refreshWaypointsList() {
            fetch('/api/get_waypoints')
                .then(r => r.json())
                .then(list => {
                    const box = document.getElementById('waypointTableBox');
                    if (!list || list.length === 0) {
                        box.innerHTML = '暂未记录点位 (拖动机械臂后点击上方按钮即可记录)';
                        return;
                    }
                    let html = '<div style="display:flex; flex-direction:column; gap:4px;">';
                    list.forEach(wp => {
                        const c = wp.cartesian_coords_mm;
                        const j = wp.joint_angles_deg;
                        html += `<div><strong>#${wp.index} ${wp.name}</strong> [${wp.timestamp}] | X:${c[0].toFixed(0)} Y:${c[1].toFixed(0)} Z:${c[2].toFixed(0)} | J:[${j.map(v=>v.toFixed(0)).join(',')}] | 夹爪:${wp.gripper_percent}%</div>`;
                    });
                    html += '</div>';
                    box.innerHTML = html;
                    box.scrollTop = box.scrollHeight;
                });
        }

        // LED 灯语
        function setLed(status) {
            fetch('/api/led', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({status: status})
            });
        }

        // 启动巡检扫描
        function startScan() {
            fetch('/api/scan', {method: 'POST'}).then(showControlResult);
        }

        // OpenVLA 动作测试
        function testOpenVLAStep(vec) {
            fetch('/api/openvla_step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action_7d: vec})
            }).then(showControlResult);
        }

        // 自然语言指令
        function sendCommand() {
            const input = document.getElementById('cmdInput');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';

            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({instruction: text})
            }).then(() => updateTelemetry());
        }

        function sendQuick(txt) {
            document.getElementById('cmdInput').value = txt;
            sendCommand();
        }

        function sendManual(action) {
            fetch('/api/manual', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: action})
            }).then(showControlResult);
        }

        // 手眼标定偏置微调
        function nudgeCalibOffset(axis, delta) {
            fetch('/api/calib_offset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({axis: axis, delta: delta})
            }).then(r => r.json()).then(() => updateTelemetry());
        }

        function resetCalibOffset() {
            fetch('/api/calib_offset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'reset'})
            }).then(r => r.json()).then(() => updateTelemetry());
        }

        // 光标悬停坐标指示
        function handleCanvasMouseMove(event) {
            const img = event.target;
            const rect = img.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;
            const normX = Math.round((clickX / rect.width) * 1000);
            const normY = Math.round((clickY / rect.height) * 1000);
            const el = document.getElementById('hoverCoords');
            if (el) {
                el.innerText = `光标: [${normX}, ${normY}]`;
            }
        }

        // 画面直接点击抓取 (Click to Pick)
        function handleCanvasClick(event) {
            const img = event.target;
            const rect = img.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;

            const normX = Math.round((clickX / rect.width) * 1000);
            const normY = Math.round((clickY / rect.height) * 1000);

            if (document.getElementById('clickMode').value === 'preview') {
                const result = document.getElementById('positionPreview');
                result.innerText = '正在计算坐标...';
                fetch('/api/preview_position', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({norm_x: normX, norm_y: normY})
                }).then(r => r.json()).then(data => {
                    result.innerText = data.success
                        ? `预览 X:${data.x_mm.toFixed(1)} Y:${data.y_mm.toFixed(1)} mm`
                        : (data.error || '坐标计算失败');
                }).catch(() => { result.innerText = '坐标预览连接失败'; });
                return;
            }

            if (confirm(`🎯 确认在点击位置 [${normX}, ${normY}] 抓取物体并搬运到工位 P5 吗？\n抓取会使用当前设定的高度和姿态，请先确认适用于棋子。`)) {
                fetch('/api/click_pick', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({norm_x: normX, norm_y: normY, target_station: 'P5'})
                }).then(r => r.json()).then(data => {
                    if (!data.success) alert(data.error || '抓取请求未执行');
                    updateTelemetry();
                }).catch(() => alert('无法连接控制大屏，请检查服务是否运行'));
            }
        }

        // 语音输入 (Web Speech API)
        let recognition = null;
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.lang = 'zh-CN';
            recognition.continuous = false;

            recognition.onresult = function(event) {
                const text = event.results[0][0].transcript;
                document.getElementById('cmdInput').value = text;
                document.getElementById('voiceBtn').classList.remove('listening');
                sendCommand();
            };
            recognition.onerror = function() {
                document.getElementById('voiceBtn').classList.remove('listening');
            };
            recognition.onend = function() {
                document.getElementById('voiceBtn').classList.remove('listening');
            };
        }

        function toggleVoice() {
            if (!recognition) {
                alert('当前浏览器暂不支持 Web Speech API，推荐使用 Edge 或 Chrome 浏览器。');
                return;
            }
            const btn = document.getElementById('voiceBtn');
            btn.classList.add('listening');
            recognition.start();
        }

        // ---------------- 象棋人机对弈交互 ----------------
        function startXqGame() {
            fetch('/api/xiangqi/start', {method: 'POST'})
                .then(r => r.json())
                .then(d => updateXiangqiStatus());
        }

        function stopXqGame() {
            fetch('/api/xiangqi/stop', {method: 'POST'})
                .then(r => r.json())
                .then(d => updateXiangqiStatus());
        }

        function resetXqGame() {
            if (confirm("确认重置象棋棋局为初始开局状态吗？")) {
                fetch('/api/xiangqi/reset', {method: 'POST'})
                    .then(r => r.json())
                    .then(d => updateXiangqiStatus());
            }
        }

        function toggleXqGrid() {
            fetch('/api/xiangqi/toggle_grid', {method: 'POST'})
                .then(r => r.json())
                .then(d => updateXiangqiStatus());
        }

        function toggleXqFocus() {
            fetch('/api/xiangqi/toggle_focus', {method: 'POST'})
                .then(r => r.json())
                .then(d => updateXiangqiStatus());
        }

        function autoLocateXqBoard() {
            const btn = document.getElementById('btnXqAutoLocate');
            const statusEl = document.getElementById('xqLocateStatus');
            if (btn) btn.disabled = true;
            if (statusEl) {
                statusEl.style.display = 'block';
                statusEl.innerText = '⏳ 正在调用相机与 Qwen3.8-Flash 大模型进行高精棋盘定位，请稍候约 3~5 秒...';
            }
            fetch('/api/xiangqi/auto_locate', {method: 'POST'})
                .then(r => r.json())
                .then(d => {
                    if (btn) btn.disabled = false;
                    if (d.success) {
                        if (statusEl) {
                            statusEl.innerText = '✅ 棋盘四角定位成功！网格已实时更新。';
                            setTimeout(() => { statusEl.style.display = 'none'; }, 4000);
                        }
                    } else {
                        if (statusEl) {
                            statusEl.innerText = '❌ 定位失败: ' + (d.error || '未知错误');
                        }
                    }
                    updateXiangqiStatus();
                })
                .catch(err => {
                    if (btn) btn.disabled = false;
                    if (statusEl) statusEl.innerText = '❌ 网络请求异常: ' + err;
                });
        }

        function scanAndStartEndgame() {
            const btn = document.getElementById('btnXqEndgame');
            const statusEl = document.getElementById('xqEndgameStatus');
            if (btn) btn.disabled = true;
            if (statusEl) {
                statusEl.style.display = 'block';
                statusEl.innerText = '🧩 正在拍摄当前残局画面，调用大模型分析所有存留棋子并推演局面 FEN...';
            }
            fetch('/api/xiangqi/endgame_detect', {method: 'POST'})
                .then(r => r.json())
                .then(d => {
                    if (btn) btn.disabled = false;
                    if (d.success) {
                        if (statusEl) {
                            statusEl.innerText = `✅ 残局识别成功！识别出 ${d.piece_count} 颗棋子，已载入局面并进入轻量对弈监控！`;
                            setTimeout(() => { statusEl.style.display = 'none'; }, 6000);
                        }
                    } else {
                        if (statusEl) {
                            statusEl.innerText = '❌ 残局识别失败: ' + (d.error || '未知错误');
                        }
                    }
                    updateXiangqiStatus();
                })
                .catch(err => {
                    if (btn) btn.disabled = false;
                    if (statusEl) statusEl.innerText = '❌ 请求异常: ' + err;
                });
        }

        function updateXiangqiStatus() {
            fetch('/api/xiangqi/status')
                .then(r => r.json())
                .then(d => {
                    const elState = document.getElementById('xqState');
                    if (elState) elState.innerText = d.state;
                    const elText = document.getElementById('xqStatusText');
                    if (elText) elText.innerText = d.status_text;
                    const elTurn = document.getElementById('xqTurn');
                    if (elTurn) elTurn.innerHTML = d.turn === 'w' ? '<span style="color:#ef4444; font-weight:bold;">红方 (人类玩家)</span>' : '<span style="color:#38bdf8; font-weight:bold;">黑方 (Pikafish AI)</span>';
                    const elFen = document.getElementById('xqFen');
                    if (elFen) elFen.innerText = d.fen;
                    const elLast = document.getElementById('xqLastMove');
                    if (elLast) elLast.innerText = `红: ${d.last_red_move || '--'} | 黑: ${d.last_black_move || '--'}`;
                    const elOffset = document.getElementById('xqOffsetSummary');
                    if (elOffset && d.calib_offset) {
                        const fmt = v => `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(1)}`;
                        elOffset.innerText = `ΔX:${fmt(d.calib_offset.x)} ΔY:${fmt(d.calib_offset.y)} ΔZ:${fmt(d.calib_offset.z)} mm`;
                    }
                    const elRun = document.getElementById('btnXqStart');
                    const elStop = document.getElementById('btnXqStop');
                    if (elRun && elStop) {
                        if (d.is_running) {
                            elRun.style.display = 'none';
                            elStop.style.display = 'block';
                        } else {
                            elRun.style.display = 'block';
                            elStop.style.display = 'none';
                        }
                    }
                    const histBox = document.getElementById('xqHistory');
                    if (histBox && d.history) {
                        let html = '';
                        d.history.forEach((h, i) => {
                            const tag = h.side === 'red' ? '<span style="color:#ef4444;">[红方]</span>' : '<span style="color:#38bdf8;">[黑方]</span>';
                            const cap = h.captured ? ` (吃 ${h.captured})` : '';
                            html += `<div style="font-size:0.75rem; padding:3px 0; border-bottom:1px solid #334155;">#${i+1} ${tag} <b>${h.move}</b>${cap} <span style="color:#64748b; font-size:0.7rem; float:right;">${h.time}</span></div>`;
                        });
                        histBox.innerHTML = html || '<div style="color:#64748b; font-size:0.75rem;">暂无走子记录，点击上方启动按钮开始对弈</div>';
                    }
                }).catch(()=>{});
        }
        setInterval(updateXiangqiStatus, 1500);

        window.addEventListener('DOMContentLoaded', () => {
            updateTelemetry();
            refreshWaypointsList();
            updateXiangqiStatus();
        });
    </script>
</body>
</html>
"""

# ==============================================================================
# RESTful 后端路由 (集成所有机械臂 Tools)
# ==============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/raw_frame')
def api_raw_frame():
    f = cam.get_latest_frame()
    if f is None:
        return ("No frame", 503)
    ret, jpeg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/api/preview_position', methods=['POST'])
def api_preview_position():
    """仅预览棋盘 X、Y 坐标，不发送机械臂或夹爪命令。"""
    data = request.get_json(silent=True) or {}
    try:
        nx, ny = float(data['norm_x']), float(data['norm_y'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'success': False, 'error': '请选择有效的画面位置'}), 400
    if not (math.isfinite(nx) and math.isfinite(ny) and 0 <= nx <= 1000 and 0 <= ny <= 1000):
        return jsonify({'success': False, 'error': '画面位置超出范围'}), 400
    frame = cam.get_latest_frame()
    if frame is None:
        return jsonify({'success': False, 'error': '摄像头画面尚未就绪'}), 503
    h, w = frame.shape[:2]
    if mapper.image_size and list(mapper.image_size) != [w, h]:
        return jsonify({'success': False, 'error': '画面尺寸与标定尺寸不一致，请重新核对'}), 400
    if mapper.matrix is None or not mapper.contains_pixel(nx, ny):
        return jsonify({'success': False, 'error': '请选择棋盘标定范围内的点'}), 400
    x_mm, y_mm, _ = mapper.pixel_to_robot(nx, ny)
    with state_lock:
        x_mm += telemetry.get('calib_offset_x', 0.0)
        y_mm += telemetry.get('calib_offset_y', 0.0)
    return jsonify({'success': True, 'x_mm': round(x_mm, 1), 'y_mm': round(y_mm, 1)})

def sanitize_for_json(obj):
    """递归将 numpy 标量与数组转换为原生 Python 类型，彻底杜绝 float32 not JSON serializable 异常"""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

@app.route('/api/status')
def api_status():
    refresh_pose()
    with state_lock:
        _sync_calib_offset_locked()
        return jsonify(sanitize_for_json(telemetry))

def reject_if_busy():
    """运动中拒绝其他控制请求，避免排队后意外执行。"""
    with state_lock:
        if telemetry.get("is_busy"):
            return jsonify({"success": False, "error": "机械臂正在执行任务，请等待结束或使用急停"}), 409
    return None

@app.route('/api/jog_cartesian', methods=['POST'])
def api_jog_cartesian():
    """笛卡尔空间微调点动"""
    blocked = reject_if_busy()
    if blocked: return blocked
    with state_lock:
        if telemetry.get("drag_teach", False):
            driver.focus_all_servos()
            telemetry["drag_teach"] = False
            add_log("🔒 自动锁定舵机扭矩，恢复刚性保持")
            time.sleep(0.2)
    data = request.json or {}
    axis = data.get("axis", "x")
    delta = float(data.get("delta", 5.0))
    res = executor.jog_cartesian(axis, delta)
    if res.get("success"):
        add_log(f"点动执行: 轴【{axis.upper()}】增量 {delta:+.1f}mm/°")
    else:
        add_log(f"点动拦截: {res.get('error')}")
    refresh_pose()
    return jsonify(res)

@app.route('/api/jog_joint', methods=['POST'])
def api_jog_joint():
    """关节角度微调点动"""
    blocked = reject_if_busy()
    if blocked: return blocked
    with state_lock:
        if telemetry.get("drag_teach", False):
            driver.focus_all_servos()
            telemetry["drag_teach"] = False
            add_log("🔒 自动锁定舵机扭矩，恢复刚性保持")
            time.sleep(0.2)
    data = request.json or {}
    joint_idx = int(data.get("joint_idx", 1))
    delta_deg = float(data.get("delta_deg", 5.0))
    res = executor.jog_joint(joint_idx, delta_deg)
    if res.get("success"):
        add_log(f"关节点动: J{joint_idx} 增量 {delta_deg:+.1f}°")
    else:
        add_log(f"关节拦截: {res.get('error')}")
    refresh_pose()
    return jsonify(res)

@app.route('/api/gripper_slider', methods=['POST'])
def api_gripper_slider():
    """连续行程夹爪控制"""
    blocked = reject_if_busy()
    if blocked: return blocked
    data = request.json or {}
    val = int(data.get("value", 100))
    res = executor.control_gripper(value_percent=val)
    refresh_pose()
    return jsonify(res)

@app.route('/api/init_gripper', methods=['POST'])
def api_init_gripper():
    """重新初始化 Modbus ID 14 Pro 自适应电动夹爪"""
    blocked = reject_if_busy()
    if blocked: return blocked
    ok = driver.init_gripper(14)
    val = driver.get_gripper_value()
    telemetry["gripper_val"] = val
    if ok:
        add_log(f"🤏 Pro 自适应电动夹爪 (Modbus ID:14) 初始化成功，当前开度: {val}%")
    else:
        add_log(f"⚠️ Pro 夹爪已执行初始化，当前开度: {val}%")
    return jsonify({"success": ok, "gripper_val": val})

@app.route('/api/move_station', methods=['POST'])
def api_move_station():
    """巡航至预设工位"""
    blocked = reject_if_busy()
    if blocked: return blocked
    with state_lock:
        if telemetry.get("drag_teach", False):
            driver.focus_all_servos()
            telemetry["drag_teach"] = False
            add_log("🔒 自动锁定舵机扭矩，恢复刚性保持")
            time.sleep(0.2)
    data = request.json or {}
    st = data.get("station", "HOME")
    add_log(f"快速巡航至工位: 【{st}】")
    res = executor.move_to_station(st)
    refresh_pose()
    return jsonify(res)

@app.route('/api/drag_teach', methods=['POST'])
def api_drag_teach():
    """拖拽示教模式开关"""
    blocked = reject_if_busy()
    if blocked: return blocked
    data = request.json or {}
    enable = bool(data.get("enable", False))
    res = executor.set_drag_teach(enable)
    telemetry["drag_teach"] = enable
    add_log(res.get("msg", ""))
    return jsonify(res)

@app.route('/api/record_waypoint', methods=['POST'])
def api_record_waypoint():
    """录制当前点位"""
    blocked = reject_if_busy()
    if blocked: return blocked
    refresh_pose()
    idx = len(recorded_waypoints) + 1
    req_name = (request.json or {}).get("name")
    name = req_name if req_name else f"P{idx}"
    coords = list(telemetry["coords"])
    angles = list(telemetry["angles"])
    grip = int(telemetry["gripper_val"])
    
    wp = {
        "index": idx,
        "name": name,
        "timestamp": time.strftime("%H:%M:%S"),
        "joint_angles_deg": angles,
        "cartesian_coords_mm": coords,
        "gripper_percent": grip
    }
    recorded_waypoints.append(wp)
    
    # 同步写入 latest_trajectory.json
    export_file = os.path.join(TRAJ_DIR, "latest_trajectory.json")
    try:
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump({"total": len(recorded_waypoints), "waypoints": recorded_waypoints}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    add_log(f"📍 记录点位【{name}】: X={coords[0]:.0f}, Y={coords[1]:.0f}, Z={coords[2]:.0f} | 夹爪={grip}%")
    return jsonify({"success": True, "waypoint": wp, "total": len(recorded_waypoints)})

@app.route('/api/get_waypoints', methods=['GET'])
def api_get_waypoints():
    """获取所有已记录的点位"""
    return jsonify(recorded_waypoints)

@app.route('/api/clear_waypoints', methods=['POST'])
def api_clear_waypoints():
    """清空所有点位"""
    recorded_waypoints.clear()
    add_log("🗑️ 已清空当前录制的示教轨迹点位。")
    return jsonify({"success": True})

@app.route('/api/replay_waypoints', methods=['POST'])
def api_replay_waypoints():
    """在真机上复现回放所有点位"""
    blocked = reject_if_busy()
    if blocked: return blocked
    if not recorded_waypoints:
        return jsonify({"success": False, "error": "没有可回放的点位"})

    add_log(f"🚀 启动轨迹回放: 共 {len(recorded_waypoints)} 个点位...")
    def replay_worker():
        executor.set_drag_teach(False)
        time.sleep(0.5)
        for i, wp in enumerate(recorded_waypoints):
            add_log(f"▶️ 回放 [{i+1}/{len(recorded_waypoints)}] -> 【{wp['name']}】")
            driver.set_gripper_value(wp["gripper_percent"], speed=60)
            driver.send_angles(wp["joint_angles_deg"], speed=15)
            time.sleep(0.5)
        add_log("🎉 轨迹回放执行完成！")

    threading.Thread(target=replay_worker, daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/led', methods=['POST'])
def api_led():
    """设置头部 LED 状态灯"""
    blocked = reject_if_busy()
    if blocked: return blocked
    status = (request.json or {}).get("status", "IDLE")
    res = executor.set_status_led(status)
    return jsonify(res)

@app.route('/api/scan', methods=['POST'])
def api_scan():
    """三维全景巡检扫描"""
    blocked = reject_if_busy()
    if blocked: return blocked
    add_log("🚀 启动三维多视点全景扫描巡检...")
    telemetry["current_step"] = "正在启动巡检扫描..."
    telemetry["step_progress"] = 15

    def scan_worker():
        def scan_cb(code, desc, pct=50):
            telemetry["current_step"] = f"巡检中: {desc}"
            telemetry["step_progress"] = pct
            add_log(f"航点扫描完成: {desc}")
            refresh_pose()

        executor.scan_workspace(scan_cb)
        fresh_frame = cam.get_latest_frame()
        if fresh_frame is not None:
            objs = detect_all_table_objects(fresh_frame, mapper=mapper)
            with state_lock:
                telemetry["live_detected_objects"] = objs
            if objs:
                names = "、".join([o["name"] for o in objs])
                add_log(f"🎉 全景巡检完毕！Qwen3.8-Flash 成功锁定 {len(objs)} 个工件: {names}")
            else:
                add_log("🎉 全景巡检完毕！工作台当前无明显工件。")
        else:
            add_log("🎉 全景巡检扫描工序全部执行完毕！")
        telemetry["current_step"] = "三维巡检全景扫描已圆满完成"
        telemetry["step_progress"] = 100
        refresh_pose()

    threading.Thread(target=scan_worker, daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/openvla_step', methods=['POST'])
def api_openvla_step():
    """OpenVLA 7-DoF 动作推演执行"""
    blocked = reject_if_busy()
    if blocked: return blocked
    data = request.json or {}
    act_7d = data.get("action_7d", [0.0]*7)
    res = executor.execute_openvla_action_step(act_7d)
    if res.get("success"):
        add_log(f"OpenVLA 动作步进: dXYZ={res['applied_delta'][:3]}")
    else:
        add_log(f"OpenVLA 限幅拦截: {res.get('error')}")
    refresh_pose()
    return jsonify(res)

def execute_embodied_instruction(instruction: str):
    """
    工业级全自主具身视觉指令调度引擎 (云端大模型 + 本地边缘规划双引擎闭环)
    """
    text_lower = instruction.lower()

    # 优先响应急停制动
    if any(k in text_lower for k in ["急停", "停止", "stop", "abort"]):
        executor.emergency_stop()
        add_log("🚨 紧急制动已触发！所有电机停止输出。")
        with state_lock:
            telemetry["current_step"] = "急停制动已触发"
            telemetry["step_progress"] = 100
        executor.set_status_led("ERROR")
        refresh_pose()
        return

    with state_lock:
        if telemetry.get("is_busy"):
            add_log("⚠️ 机械臂当前正忙，上一项任务尚未结束，请稍候！")
            return
        telemetry["is_busy"] = True

    # 自动解除拖拽示教并复位电机扭矩 (防止关节掉使能报警)
    with state_lock:
        if telemetry.get("drag_teach", False):
            driver.focus_all_servos()
            telemetry["drag_teach"] = False
            add_log("🔒 检测到此前处于示教释放状态，已自动上电复位舵机扭矩，恢复刚性工作！")
            time.sleep(0.3)

    try:
        telemetry["current_step"] = "正在理解具身意图与规划动作..."
        telemetry["step_progress"] = 10
        executor.set_status_led("VISION")

        # 1. 优先直连 DeepSeek-V4.1-Flash (Vision) 极速云端大模型
        cloud_result = None
        frame = cam.get_latest_frame()
        t0 = time.time()
        cloud_result = call_deepseek_embodied_vlm(instruction, frame)

        if cloud_result:
            latency = cloud_result.get("elapsed_ms", (time.time() - t0) * 1000)
            telemetry["last_latency_ms"] = latency
            telemetry["cloud_status"] = "ONLINE"
            add_log(f"☁️ Qwen3.8-Flash 极速响应完成 (耗时: {latency:.0f}ms)")
            if cloud_result.get("visual_description"):
                add_log(f"   💡 模型规划: {cloud_result['visual_description']}")
            elif cloud_result.get("reply"):
                add_log(f"   💬 模型答复: {cloud_result['reply']}")
        else:
            # 回退尝试本地 6006 端口代理
            if frame is not None:
                try:
                    b64 = frame_to_base64_optimized(frame)
                    resp = requests.post(CLOUD_VLM_URL, json={"instruction": instruction, "image_base64": b64}, timeout=5.0)
                    if resp.status_code == 200:
                        cloud_result = resp.json()
                        latency = cloud_result.get("elapsed_ms", (time.time() - t0) * 1000)
                        telemetry["last_latency_ms"] = latency
                        telemetry["cloud_status"] = "ONLINE"
                        add_log(f"☁️ 代理大模型响应完成 (耗时: {latency:.0f}ms)")
                except Exception:
                    telemetry["cloud_status"] = "OFFLINE"
                    add_log("⚡ [智能边缘引擎接管]: 云端模型离线，本地具身智能中枢已自动接管调度！")

        # 2. 自然语言问答/状态汇报意图拦截 (若为观察汇报/问答，不执行抓取动作，直接输出自然语言解答)
        if cloud_result and cloud_result.get("intent") == "chat":
            reply_msg = cloud_result.get("reply") or cloud_result.get("visual_description", "已完成画面观察。")
            add_log(f"💬 [具身大模型汇报]: {reply_msg}")
            telemetry["current_step"] = "自然语言汇报完成"
            telemetry["step_progress"] = 100
            executor.set_status_led("IDLE")
            return

        # 3. 核心意图分支判断
        # 3.1 复位待机
        if any(k in text_lower for k in ["复位", "待机", "回零", "回到原点", "初始", "home", "normal", "standby"]):
            add_log("🏠 收到复位指令，机械臂平稳返回待机姿态 NORMAL...")
            telemetry["current_step"] = "正在复位至 NORMAL..."
            telemetry["step_progress"] = 50
            executor.set_status_led("MOVING")
            executor.move_to_station("NORMAL")
            telemetry["current_step"] = "机械臂已成功复位至待机位"
            telemetry["step_progress"] = 100
            executor.set_status_led("IDLE")
            add_log("✅ 机械臂已安全到达初始待机工位 NORMAL")
            refresh_pose()
            return

        # 3.2 夹爪直接控制
        if any(k in text_lower for k in ["开爪", "张开夹爪", "打开夹爪", "松开夹爪", "释放夹爪"]):
            executor.control_gripper("OPEN")
            add_log("🤏 夹爪已按设定开度张开 (15%)")
            telemetry["current_step"] = "夹爪张开完成 (15%)"
            telemetry["step_progress"] = 100
            refresh_pose()
            return
        elif any(k in text_lower for k in ["闭爪", "合上夹爪", "关闭夹爪", "夹紧", "抓紧", "收爪"]):
            executor.control_gripper("CLOSE")
            add_log("🤏 夹爪已收爪闭合 (5%)")
            telemetry["current_step"] = "夹爪收爪完成 (5%)"
            telemetry["step_progress"] = 100
            refresh_pose()
            return

        # 3.3 挥手打招呼
        if any(k in text_lower for k in ["挥手", "打招呼", "wave"]):
            executor.set_status_led("MOVING")
            telemetry["current_step"] = "正在执行挥手动作..."
            telemetry["step_progress"] = 30
            add_log("👋 机械臂挥手互动中...")
            norm_j = [5.0, -4.0, -7.0, -57.0, 5.0, -132.0]
            driver.send_angles([norm_j[0], norm_j[1], norm_j[2], norm_j[3], norm_j[4] + 25.0, norm_j[5]], 25)
            driver.send_angles([norm_j[0], norm_j[1], norm_j[2], norm_j[3], norm_j[4] - 25.0, norm_j[5]], 25)
            driver.send_angles([norm_j[0], norm_j[1], norm_j[2], norm_j[3], norm_j[4] + 25.0, norm_j[5]], 25)
            executor.move_to_station("NORMAL")
            telemetry["current_step"] = "挥手动作完成"
            telemetry["step_progress"] = 100
            executor.set_status_led("IDLE")
            add_log("✅ 挥手动作已圆满完成")
            refresh_pose()
            return

        # 2.5 分析扫描意图与搬运意图
        need_scan = any(k in text_lower for k in ["扫视", "扫描", "巡检", "环视", "全景", "看看", "看一眼", "找找", "观察"])
        need_pick_place = any(k in text_lower for k in ["移到", "放到", "搬运", "抓取", "拿给", "送去", "放进", "转移", "移动到", "夹到", "搬到", "夹取", "抓", "放", "拿", "给"])

        # 若纯粹是巡检/扫描指令 (例如 "扫描工作台上的所有零件")，执行纯巡检工作流
        if need_scan and not need_pick_place:
            add_log("🚀 启动工作台全景三维巡检扫描...")
            def scan_cb(code, desc, pct=50):
                telemetry["current_step"] = f"全景巡检中: {desc}"
                telemetry["step_progress"] = pct
                add_log(f"航点扫描完成: {desc}")
                refresh_pose()

            executor.scan_workspace(scan_cb)
            fresh_frame = cam.get_latest_frame()
            all_objs = detect_all_table_objects(fresh_frame, mapper=mapper) if fresh_frame is not None else []
            with state_lock:
                telemetry["live_detected_objects"] = all_objs
                if all_objs:
                    telemetry["last_bbox"] = all_objs[0]["bbox"]
                    telemetry["last_detected_obj"] = all_objs[0]["name"]
                    telemetry["last_coords"] = all_objs[0]["robot_xyz"]
            telemetry["current_step"] = "全景巡检工序全部执行完毕"
            telemetry["step_progress"] = 100
            executor.set_status_led("IDLE")
            if all_objs:
                summary_str = "、".join([f"{o['name']}" for o in all_objs])
                add_log(f"🎉 全景巡检完成！Qwen3.8-Flash 成功识别到 {len(all_objs)} 个工件: {summary_str}")
            else:
                add_log("🎉 全景巡检完成！工作台视野未检测到明显工件。")
            refresh_pose()
            return

        # 1. 灵活解析目标目的地 (解除固定工位限制，支持方位、自定义坐标、预设工位)
        target_station = "P5"
        custom_dest_coord = None
        dest_desc = "工位【P5】"

        if any(k in instruction for k in ["右边", "右侧", "右方"]):
            custom_dest_coord = [280.0, -180.0, mapper.table_z + 8.0]
            dest_desc = "【右侧工作区】"
        elif any(k in instruction for k in ["左边", "左侧", "左方"]):
            custom_dest_coord = [280.0, 180.0, mapper.table_z + 8.0]
            dest_desc = "【左侧工作区】"
        elif any(k in instruction for k in ["中间", "中心", "正中"]):
            custom_dest_coord = [320.0, 0.0, mapper.table_z + 8.0]
            dest_desc = "【中央工作区】"
        elif any(k in instruction for k in ["前方", "远端", "上面"]):
            custom_dest_coord = [380.0, 0.0, mapper.table_z + 8.0]
            dest_desc = "【远端工作区】"
        else:
            m_station = re.search(r"\b(p[1-5]|normal|home)\b", text_lower)
            if m_station:
                target_station = m_station.group(1).upper()
                dest_desc = f"工位【{target_station}】"
            else:
                m_cn = re.search(r"工位([1-5])", text_lower)
                if m_cn:
                    target_station = f"P{m_cn.group(1)}"
                    dest_desc = f"工位【{target_station}】"

        final_dest = custom_dest_coord if custom_dest_coord else target_station

        # 2. 提取目标工件类别 (全面支持象棋红黑棋子、各类工件与开放实体)
        target_obj = "工件"
        if "红" in instruction:
            target_obj = "红色棋子"
        elif "黑" in instruction:
            target_obj = "黑色棋子"
        elif cloud_result and cloud_result.get("arguments", {}).get("target_object"):
            target_obj = cloud_result["arguments"]["target_object"]
        else:
            m_obj = re.search(r"(车|車|马|馬|相|象|士|仕|将|帥|帅|炮|兵|卒|棋子|盒子|急停盒|急停|黄盒|纸盒|箱子|箱|胶带|圆筒|圆环|杯子|方块|积木|尺子|笔|工件|零件|物料|物体)", instruction)
            if m_obj:
                target_obj = m_obj.group(1)

        # 3. 循环/批量指令识别 (支持“全部”、“所有”、“循环识别4个”等自动化批量作业)
        is_batch = any(k in instruction for k in ["全部", "所有", "都", "每个", "循环", "一个个", "批量", "一并", "统统", "逐个"])
        max_loops = 1
        if is_batch:
            m_num = re.search(r"(\d+)\s*个", instruction)
            if m_num:
                max_loops = int(m_num.group(1))
            elif any(k in instruction for k in ["4", "四"]):
                max_loops = 4
            elif any(k in instruction for k in ["3", "三"]):
                max_loops = 3
            elif any(k in instruction for k in ["2", "两", "二"]):
                max_loops = 2
            else:
                max_loops = 5  # 默认批量上限 5 个，防止死循环

        # 4. 解析运动速度 (支持UI滑块设定或语音/文本指定 "速度30", "快速", "慢速")
        motion_speed = telemetry.get("motion_speed", 20)
        m_spd = re.search(r"速度\s*[:=为]?\s*(\d+)", instruction)
        if m_spd:
            motion_speed = max(5, min(80, int(m_spd.group(1))))
        elif any(k in instruction for k in ["快速", "高速", "快一点", "加快", "疾速"]):
            motion_speed = 40
        elif any(k in instruction for k in ["慢速", "慢一点", "减速", "精细", "缓速"]):
            motion_speed = 10

        add_log(f"🧠 具身任务规划: [目标工件=【{target_obj}】] -> [目标目的地={dest_desc}] -> [速度={motion_speed}%] -> [批量循环={is_batch} (上限 {max_loops} 次)]")

        # 阶段 1: 全景三维扫视巡检 (可选)
        if need_scan:
            add_log("🚀 [工序 1/3]: 启动工作台多视点全景扫描巡检...")
            def scan_cb(code, desc, pct=50):
                telemetry["current_step"] = f"全景巡检中: {desc}"
                telemetry["step_progress"] = int(pct * 0.35)
                add_log(f"航点扫描完成: {desc}")
                refresh_pose()
            executor.scan_workspace(scan_cb)
            add_log("🎉 全景巡检扫描完成！机械臂已回到最佳俯视观测位。")
            time.sleep(0.5)

        with state_lock:
            off_x = telemetry.get("calib_offset_x", 0.0)
            off_y = telemetry.get("calib_offset_y", 0.0)
            off_z = telemetry.get("calib_offset_z", 0.0)

        # 阶段 2: 循环视觉识别与闭环批量搬运
        def progress_cb(step_id, desc, pct):
            telemetry["current_step"] = f"{step_id}: {desc}"
            telemetry["step_progress"] = pct
            add_log(f"   [{step_id}] -> {desc}")
            refresh_pose()

        success_count = 0
        red_chess_names = ["帅", "帥", "仕", "士", "相", "象", "车", "車", "马", "馬", "炮", "兵", "红"]
        black_chess_names = ["将", "將", "士", "象", "车", "車", "马", "馬", "炮", "卒", "黑"]

        for loop_idx in range(max_loops):
            # 获取最新鲜的工作台实时图像
            fresh_frame = cam.get_latest_frame()
            if fresh_frame is None:
                time.sleep(0.5)
                fresh_frame = cam.get_latest_frame()

            if is_batch:
                add_log(f"🔍 [循环扫描 {loop_idx+1}/{max_loops}]: 正在检测工作台上的【{target_obj}】...")
                telemetry["current_step"] = f"循环识别第 {loop_idx+1}/{max_loops} 批工件..."
                telemetry["step_progress"] = int((loop_idx / max_loops) * 100)

            all_objs = detect_all_table_objects(fresh_frame, mapper=mapper) if fresh_frame is not None else []
            
            # 筛选符合当前目标的所有候选工件
            matching_objs = []
            for o in all_objs:
                oname = o.get("name", "")
                if "红" in target_obj:
                    # 匹配所有红方棋子 (含“红”字，或纯红棋字样：兵/帅/相/仕) 且排除黑棋
                    is_red = ("红" in oname or any(k in oname for k in ["帅", "帥", "兵", "相", "仕"])) and ("黑" not in oname and "卒" not in oname and "将" not in oname)
                    if is_red:
                        matching_objs.append(o)
                elif "黑" in target_obj:
                    # 匹配所有黑方棋子 (含“黑”字，或纯黑棋字样：卒/将/士/象) 且排除红棋
                    is_black = ("黑" in oname or any(k in oname for k in ["将", "將", "卒", "士", "象"])) and ("红" not in oname and "兵" not in oname and "帅" not in oname)
                    if is_black:
                        matching_objs.append(o)
                elif target_obj in oname or oname in target_obj:
                    matching_objs.append(o)

            if not matching_objs:
                if is_batch:
                    if success_count > 0:
                        add_log(f"🏁 视觉扫描确认：视野内已无剩余【{target_obj}】，循环批处理全部完成！共搬运 {success_count} 个。")
                        telemetry["current_step"] = f"批量搬运圆满完成，共移送 {success_count} 个工件"
                    else:
                        add_log(f"⚠️ 视野内未检测到【{target_obj}】，动作安全终止。")
                        telemetry["current_step"] = f"未检测到【{target_obj}】"
                else:
                    add_log(f"❌ 工作台上未能识别到【{target_obj}】，动作安全取消！")
                    telemetry["current_step"] = f"未找到目标【{target_obj}】"
                break

            # 抓取首个匹配工件
            cur_item = matching_objs[0]
            bbox = cur_item["bbox"]
            item_name = cur_item.get("name", target_obj)

            # 阵列防撞偏置：如果是批量搬运，每个工件错开 32mm 整齐排布，避免重复堆叠碰撞
            place_offset = (0.0, success_count * 32.0, 0.0) if is_batch else None

            add_log(f"🎯 [目标锁定]: 捕获【{item_name}】(检测框 {bbox})，开始执行抓取...")
            telemetry["last_bbox"] = bbox
            telemetry["last_detected_obj"] = item_name

            executor.set_status_led("MOVING")
            res = executor.execute_visual_pick_and_place(
                item_name, bbox, final_dest,
                speed=motion_speed,
                step_callback=progress_cb,
                offset_trim=(off_x, off_y, off_z),
                place_offset=place_offset
            )

            if res.get("success"):
                success_count += 1
                add_log(f"🎉 成功移送【{item_name}】至 {dest_desc}！(累计完成: {success_count}/{max_loops})")
            else:
                add_log(f"⚠️ 搬运【{item_name}】受阻: {res.get('error')}")
                if not is_batch:
                    break

            time.sleep(0.8)

        if success_count > 0:
            telemetry["current_step"] = f"工序圆满完成！已搬运 {success_count} 个【{target_obj}】"
            telemetry["step_progress"] = 100
            add_log(f"🏆 【具身智能任务完成】共成功搬运 {success_count} 个【{target_obj}】至 {dest_desc}！")
        executor.set_status_led("IDLE")
        refresh_pose()

    except Exception as e:
        import traceback
        traceback.print_exc()
        add_log(f"❌ 任务调度执行异常: {e}")
        telemetry["current_step"] = f"异常: {e}"
        telemetry["step_progress"] = 100
        executor.set_status_led("ERROR")
    finally:
        with state_lock:
            telemetry["is_busy"] = False

@app.route('/api/command', methods=['POST'])
def api_command():
    """多模态具身视觉指令处理接口 (异步非阻塞)"""
    data = request.json or {}
    instruction = data.get("instruction", "").strip()
    if not instruction:
        return jsonify({"success": False, "error": "空指令"})

    add_log(f"用户指令: \"{instruction}\"")
    telemetry["current_step"] = "指令已受理，正在启动具身任务引擎..."
    telemetry["step_progress"] = 5
    executor.set_status_led("VISION")

    threading.Thread(target=execute_embodied_instruction, args=(instruction,), daemon=True).start()
    return jsonify({"success": True, "msg": "指令已受理，正在后台执行"})

@app.route('/api/click_pick', methods=['POST'])
def api_click_pick():
    """画面直接点击抓取 (支持智能磁吸对齐工件中心 + 偏置微调)"""
    with state_lock:
        if telemetry.get("is_busy"):
            return jsonify({"success": False, "error": "当前机械臂正在执行其他任务，请稍候！"})
        if not driver.is_connected:
            return jsonify({"success": False, "error": "当前未连接真机，画面点击不会驱动机械臂"}), 503
        stop_event = getattr(getattr(driver, "backend", driver), "stop_requested", None)
        if stop_event is not None and stop_event.is_set():
            return jsonify({"success": False, "error": "已请求停止，请检查机械臂并重新启动控制程序"}), 409
        telemetry["is_busy"] = True

    # 自动解除拖拽示教并复位电机扭矩 (防止掉使能报错)
    with state_lock:
        if telemetry.get("drag_teach", False):
            driver.focus_all_servos()
            telemetry["drag_teach"] = False
            add_log("🔒 自动锁定舵机扭矩，退出示教模式")
            time.sleep(0.3)

    try:
        data = request.json or {}
        nx = float(data.get("norm_x", 500))
        ny = float(data.get("norm_y", 500))
        station = data.get("target_station", "P5")

        # 1. 智能磁吸对齐：若点击点附近 80px 范围内有实时识别的目标工件，自动对齐到工件中心
        snapped_name = "点击工件"
        best_dist = 80.0
        snapped_bbox = None

        with state_lock:
            live_objs = list(telemetry.get("live_detected_objects", []))
            off_x = telemetry.get("calib_offset_x", 0.0)
            off_y = telemetry.get("calib_offset_y", 0.0)
            off_z = telemetry.get("calib_offset_z", 0.0)

        for o in live_objs:
            obbox = o.get("bbox")
            if obbox:
                oymin, oxmin, oymax, oxmax = obbox
                ocx = (oxmin + oxmax) / 2.0
                ocy = (oymin + oymax) / 2.0
                dist = math.hypot(nx - ocx, ny - ocy)
                if dist < best_dist:
                    best_dist = dist
                    nx, ny = ocx, ocy
                    snapped_name = o.get("name", "工件")
                    snapped_bbox = obbox

        if snapped_bbox:
            bbox = snapped_bbox
        else:
            bbox = [int(max(0, ny - 45)), int(max(0, nx - 45)), int(min(1000, ny + 45)), int(min(1000, nx + 45))]

        # 智能霍夫圆几何对齐微调：若点击处是圆形工件/棋子，精细锁定真实物理圆心
        fresh_frame = cam.get_latest_frame()
        if fresh_frame is not None:
            norm_cx, norm_cy, refined_box = refine_piece_center_circle(fresh_frame, bbox)
            bbox = refined_box
            nx, ny = norm_cx, norm_cy

        raw_xyz = mapper.pixel_to_robot(nx, ny)
        xyz = (round(raw_xyz[0] + off_x, 1), round(raw_xyz[1] + off_y, 1), round(raw_xyz[2] + off_z, 1))

        telemetry["last_bbox"] = bbox
        telemetry["last_detected_obj"] = snapped_name
        telemetry["last_coords"] = xyz
        add_log(f"🎯 点击视觉抓取: 图像点 [{nx:.0f}, {ny:.0f}] (吸附={snapped_name}) -> 物理坐标 {xyz}")

        telemetry["current_step"] = f"正在搬运【{snapped_name}】至工位 {station}..."
        telemetry["step_progress"] = 20
        executor.set_status_led("MOVING")

        def progress_cb(step_id, desc, pct):
            telemetry["current_step"] = f"{step_id}: {desc}"
            telemetry["step_progress"] = pct

        def async_click_worker():
            try:
                curr_speed = telemetry.get("motion_speed", 20)
                result = executor.execute_visual_pick_and_place(
                    snapped_name, bbox, station, 
                    speed=curr_speed,
                    step_callback=progress_cb,
                    offset_trim=(off_x, off_y, off_z)
                )
                if result.get("success"):
                    telemetry["current_step"] = f"【{snapped_name}】抓取搬运圆满完成"
                    executor.set_status_led("IDLE")
                    add_log(f"🎉 点击抓取工序圆满完成！【{snapped_name}】已成功送达工位【{station}】！")
                else:
                    reason = result.get("error", "抓取未执行")
                    telemetry["current_step"] = f"抓取受阻: {reason}"
                    executor.set_status_led("ERROR")
                    add_log(f"⚠️ 抓取受阻: {reason}")
                telemetry["step_progress"] = 100
            except Exception as e:
                add_log(f"❌ 点击抓取执行异常: {e}")
                telemetry["current_step"] = f"异常: {e}"
                telemetry["step_progress"] = 100
                executor.set_status_led("ERROR")
            finally:
                with state_lock:
                    telemetry["is_busy"] = False
                refresh_pose()

        threading.Thread(target=async_click_worker, daemon=True).start()
        return jsonify({"success": True})
    except Exception as e:
        with state_lock:
            telemetry["is_busy"] = False
        add_log(f"❌ 点击抓取准备异常: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/set_speed', methods=['POST'])
def api_set_speed():
    """实时调节机械臂全局运动速度 (5~80%)"""
    data = request.json or {}
    speed = int(data.get("speed", 20))
    speed = max(5, min(80, speed))
    with state_lock:
        telemetry["motion_speed"] = speed
    add_log(f"⚡ 机械臂运动速度已调节为: {speed}%")
    return jsonify({"success": True, "speed": speed})

@app.route('/api/calib_offset', methods=['POST'])
def api_calib_offset():
    """手眼标定实时微调偏置接口 (dx, dy, dz)，同步持久化到标定文件并供象棋模块读取"""
    data = request.json or {}
    axis = data.get("axis", "").lower()
    delta = float(data.get("delta", 0.0))
    action = data.get("action")

    with state_lock:
        changed = False
        if action == "reset":
            telemetry["calib_offset_x"] = 0.0
            telemetry["calib_offset_y"] = 0.0
            telemetry["calib_offset_z"] = 0.0
            changed = True
            add_log("🔄 视觉手眼偏置已重置清零 (ΔX=0, ΔY=0, ΔZ=0)")
        elif axis in ["x", "y", "z"]:
            key = f"calib_offset_{axis}"
            telemetry[key] = round(telemetry.get(key, 0.0) + delta, 1)
            changed = True
            add_log(f"🎯 视觉手眼偏置微调: Δ{axis.upper()} = {telemetry[key]:+.1f}mm")

        if changed:
            mapper.set_offset(
                telemetry.get("calib_offset_x", 0.0),
                telemetry.get("calib_offset_y", 0.0),
                telemetry.get("calib_offset_z", 0.0),
                persist=True,
            )

        res = {
            "success": True,
            "calib_offset_x": telemetry.get("calib_offset_x", 0.0),
            "calib_offset_y": telemetry.get("calib_offset_y", 0.0),
            "calib_offset_z": telemetry.get("calib_offset_z", 0.0)
        }
    return jsonify(res)

@app.route('/api/manual', methods=['POST'])
def api_manual():
    act = (request.json or {}).get("action")
    if act == "emergency_stop":
        executor.emergency_stop()
        add_log("🚨 已发送停止请求，请确认机械臂已停止。")
    elif act == "open_gripper":
        blocked = reject_if_busy()
        if blocked: return blocked
        executor.control_gripper("OPEN")
        add_log("夹爪全开")
    elif act == "close_gripper":
        blocked = reject_if_busy()
        if blocked: return blocked
        executor.control_gripper("CLOSE")
        add_log("夹爪闭紧")
    refresh_pose()
    return jsonify({"success": True})

# ------------------------------------------------------------------------------
# 象棋人机对弈闭环专用 API
# ------------------------------------------------------------------------------
@app.route('/api/xiangqi/status')
def api_xiangqi_status():
    ox, oy, oz = get_calib_offset()
    with xiangqi_agent.lock:
        return jsonify({
            "is_running": xiangqi_agent.is_running,
            "state": xiangqi_agent.current_state,
            "status_text": xiangqi_agent.status_text,
            "turn": xiangqi_agent.board.turn,
            "fen": xiangqi_agent.board.get_fen(),
            "last_red_move": xiangqi_agent.last_red_move,
            "last_black_move": xiangqi_agent.last_black_move,
            "history": xiangqi_agent.move_history,
            "show_grid": telemetry.get("show_xiangqi_grid", True),
            "corners_norm": xiangqi_grid_mapper.corners_norm,
            "calib_offset": {"x": ox, "y": oy, "z": oz}
        })

@app.route('/api/xiangqi/start', methods=['POST'])
def api_xiangqi_start():
    xiangqi_agent.start_game()
    with state_lock:
        telemetry["show_xiangqi_grid"] = True
    add_log("🚀 启动中国象棋具身智能对弈闭环！机械臂平稳复位至待机位采样基准图...")
    return jsonify({"success": True})

@app.route('/api/xiangqi/stop', methods=['POST'])
def api_xiangqi_stop():
    xiangqi_agent.stop_game()
    add_log("⏸️ 暂停象棋对弈闭环")
    return jsonify({"success": True})

@app.route('/api/xiangqi/reset', methods=['POST'])
def api_xiangqi_reset():
    xiangqi_agent.stop_game()
    with xiangqi_agent.lock:
        xiangqi_agent.board.set_fen(INITIAL_FEN)
        xiangqi_agent.move_history.clear()
        xiangqi_agent.last_red_move = None
        xiangqi_agent.last_black_move = None
        xiangqi_agent.current_state = "IDLE"
        xiangqi_agent.status_text = "棋盘已重置为初始开局状态"
    add_log("🔄 象棋棋局已重置为初始开局状态 (INITIAL FEN)")
    return jsonify({"success": True})

@app.route('/api/xiangqi/endgame_detect', methods=['POST'])
def api_xiangqi_endgame_detect():
    """残局模式：一键识别当前非初始棋盘上的所有棋子并启动对弈"""
    try:
        from xiangqi_endgame import detect_endgame_from_image
        # 1. 机械臂先平稳移至 NORMAL 待机位，避免遮挡
        executor.move_to_station("HOME")
        time.sleep(1.2)

        # 2. 获取干净相机帧
        frame = cam.get_latest_frame()
        if frame is None:
            return jsonify({"success": False, "error": "摄像头尚未就绪"}), 503

        # 3. 运行残局视觉识别
        res = detect_endgame_from_image(frame, xiangqi_grid_mapper)
        if not res.get("success"):
            return jsonify({"success": False, "error": "残局大模型识别失败"}), 500

        fen = res.get("fen")
        is_rotated = res.get("is_rotated", False)
        piece_dict = res.get("piece_dict", {})

        # 4. 载入残局并启动对弈
        xiangqi_agent.start_game(fen=fen, is_rotated=is_rotated, is_endgame=True)
        with state_lock:
            telemetry["show_xiangqi_grid"] = True

        add_log(f"🧩 【残局模式已就绪】成功识别 {len(piece_dict)} 颗棋子！视角对齐: {'反向(180°)' if is_rotated else '正向'}，FEN: {fen}")
        return jsonify(sanitize_for_json({
            "success": True,
            "fen": fen,
            "piece_count": len(piece_dict),
            "pieces": piece_dict,
            "is_rotated": is_rotated
        }))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/xiangqi/auto_locate', methods=['POST'])
def api_xiangqi_auto_locate():
    """一键调用视觉模型与 OpenCV 亚像素角点对齐空棋盘"""
    try:
        from xiangqi_auto_locate import run_one_click_board_localization
        res = run_one_click_board_localization()
        if res.get("success"):
            corners = res.get("corners", {})
            with state_lock:
                telemetry["show_xiangqi_grid"] = True
            add_log("🎯 棋盘一键视觉标定完成，已自动更新 4 角点并重新投影 9x10 象棋网格！")
            return jsonify(sanitize_for_json({"success": True, "corners": corners}))
        else:
            return jsonify({"success": False, "error": res.get("error", "标定未完全成功")}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/xiangqi/toggle_grid', methods=['POST'])
def api_xiangqi_toggle_grid():
    with state_lock:
        telemetry["show_xiangqi_grid"] = not telemetry.get("show_xiangqi_grid", False)
        cur = telemetry["show_xiangqi_grid"]
    return jsonify({"success": True, "show_grid": cur})

@app.route('/api/xiangqi/toggle_focus', methods=['POST'])
def api_xiangqi_toggle_focus():
    with state_lock:
        telemetry["focus_xiangqi_board"] = not telemetry.get("focus_xiangqi_board", False)
        cur = telemetry["focus_xiangqi_board"]
    return jsonify({"success": True, "focus_board": cur})

@app.route('/api/xiangqi/set_corners', methods=['POST'])
def api_xiangqi_set_corners():
    data = request.json or {}
    a0 = data.get("a0")
    i0 = data.get("i0")
    i9 = data.get("i9")
    a9 = data.get("a9")
    if a0 and i0 and i9 and a9:
        xiangqi_grid_mapper.set_corners(a0, i0, i9, a9)
        add_log(f"📐 象棋 4 角点已更新: a0={a0}, i0={i0}, i9={i9}, a9={a9}")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "坐标参数不足"}), 400

def kill_stale_process_on_port(port: int = 5000):
    """自动清理占用 5000 端口的僵尸进程，彻底杜绝 Errno 10048 崩溃"""
    try:
        import subprocess, re
        out = subprocess.check_output("netstat -ano -p tcp", shell=True).decode()
        current_pid = os.getpid()
        for line in out.strip().splitlines():
            m = re.search(rf":{port}\s+.*LISTENING\s+(\d+)", line)
            if m:
                pid = int(m.group(1))
                if pid != current_pid and pid > 0:
                    print(f"⚠️ 发现端口 {port} 被旧后台进程 (PID: {pid}) 占用，正在自动释放...")
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                    time.sleep(0.8)
    except Exception:
        pass

def cleanup_and_exit(signum=None, frame=None):
    print("\n🛑 正在安全关闭 Web Cockpit 进程与外设连接...")
    try:
        cam.release()
    except Exception:
        pass
    try:
        driver.focus_all_servos()
    except Exception:
        pass
    os._exit(0)

try:
    import signal
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
except Exception:
    pass

def start_server():
    kill_stale_process_on_port(5000)
    # 启动底层机械臂状态 2Hz 稳态遥测守护线程 (防阻塞、零排队)
    threading.Thread(target=pose_telemetry_loop, daemon=True).start()
    # 启动 Qwen3-VL 异步具身感知后台守护进程 (3.5s 周期，轻负载)
    threading.Thread(target=qwen_live_perception_loop, daemon=True).start()
    
    # 自动在默认浏览器中打开控制台
    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 65)
    print("  🚀 FlexArm 具身视觉操控大屏与工具箱已在 http://127.0.0.1:5000 启动")
    print("=" * 65)
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    start_server()
