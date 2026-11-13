"""按设备身份选择机械臂的工作台摄像头。"""

import json
import sys
from pathlib import Path

import cv2
from cv2_enumerate_cameras import enumerate_cameras


CONFIG_FILE = Path(__file__).resolve().with_name("camera_config.json")


def select_robot_camera(config):
    """枚举设备元数据，并匹配已配置的外接摄像头。"""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    expected_name = config["device_name"].strip().casefold()
    expected_vid = int(config["usb_vid_hex"], 16)
    expected_pid = int(config["usb_pid_hex"], 16)
    cameras = enumerate_cameras(backend)
    matches = [
        camera for camera in cameras
        if camera.name.strip().casefold() == expected_name
        and camera.vid == expected_vid
        and camera.pid == expected_pid
    ]
    expected_path = config.get("device_path", "").casefold()
    for camera in matches:
        if expected_path and camera.path.casefold() == expected_path:
            return camera
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"未检测到外接摄像头 {config['device_name']} "
            f"({expected_vid:04X}:{expected_pid:04X})，请检查 USB 连接"
        )
    raise RuntimeError("检测到多个同型号外接摄像头，请在 camera_config.json 中核对 device_path")


def open_robot_camera(config_path=CONFIG_FILE):
    """解析当前设备编号并打开配置中的摄像头，返回 OpenCV 采集对象。"""
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    camera = select_robot_camera(config)
    cap = cv2.VideoCapture(camera.index, camera.backend)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开外接摄像头 {camera.name}，请检查设备是否被其他程序占用")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width_px"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height_px"])
        cap.set(cv2.CAP_PROP_FPS, config["fps"])
    except Exception:
        cap.release()
        raise
    print(f"📷 已连接外接摄像头 {camera.name}，当前设备编号 {camera.index}")
    return cap
