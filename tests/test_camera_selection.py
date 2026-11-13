"""离线验证外接摄像头选择，覆盖设备编号变化和设备缺失。"""

import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from camera_device import CONFIG_FILE, open_robot_camera, select_robot_camera
from robot_vision_client import CameraStreamThread


ROOT = Path(__file__).resolve().parents[1]


def make_camera(index, name, vid, pid, path):
    return SimpleNamespace(
        index=index, name=name, vid=vid, pid=pid,
        path=path, backend=cv2.CAP_DSHOW,
    )


class CameraSelectionTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        self.external = make_camera(
            1, "USB 2.0 Camera", 0x0BDA, 0x5A63, self.config["device_path"],
        )
        self.internal = make_camera(
            0, "Integrated Camera", 0x30C9, 0x00A6, "internal-camera-path",
        )

    def test_selects_external_device(self):
        with patch("camera_device.enumerate_cameras", return_value=[self.internal, self.external]):
            self.assertIs(select_robot_camera(self.config), self.external)

    def test_external_index_can_change_to_zero(self):
        self.internal.index = 1
        self.external.index = 0
        with patch("camera_device.enumerate_cameras", return_value=[self.external, self.internal]), \
                patch("camera_device.cv2.VideoCapture") as capture:
            cap = open_robot_camera()
        capture.assert_called_once_with(0, cv2.CAP_DSHOW)
        cap.release()

    def test_replugged_unique_device_can_have_new_path(self):
        self.external.path = "new-usb-port-path"
        with patch("camera_device.enumerate_cameras", return_value=[self.internal, self.external]):
            self.assertIs(select_robot_camera(self.config), self.external)

    def test_missing_external_never_opens_a_camera(self):
        with patch("camera_device.enumerate_cameras", return_value=[self.internal]), \
                patch("camera_device.cv2.VideoCapture") as capture, \
                self.assertRaisesRegex(RuntimeError, "未检测到外接摄像头"):
            open_robot_camera()
        capture.assert_not_called()

    def test_saved_path_disambiguates_same_model(self):
        other = copy.copy(self.external)
        other.index = 2
        other.path = "other-camera-path"
        with patch("camera_device.enumerate_cameras", return_value=[other, self.external]):
            self.assertIs(select_robot_camera(self.config), self.external)

    def test_ambiguous_external_never_opens_a_camera(self):
        other = copy.copy(self.external)
        self.external.path = "new-path-1"
        other.path = "new-path-2"
        with patch("camera_device.enumerate_cameras", return_value=[self.external, other]), \
                patch("camera_device.cv2.VideoCapture") as capture, \
                self.assertRaisesRegex(RuntimeError, "多个同型号"):
            open_robot_camera()
        capture.assert_not_called()

    def test_failed_open_releases_capture(self):
        with patch("camera_device.enumerate_cameras", return_value=[self.external]), \
                patch("camera_device.cv2.VideoCapture") as capture:
            capture.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "无法打开外接摄像头"):
                open_robot_camera()
        capture.return_value.release.assert_called_once()

    def test_stream_default_uses_external_selection(self):
        with patch("robot_vision_client.open_robot_camera") as open_camera, \
                patch("robot_vision_client.threading.Thread"):
            stream = CameraStreamThread()
            self.assertIs(stream.cap, open_camera.return_value)
            open_camera.assert_called_once_with()
            stream.release()

    def test_program_entry_points_use_default_selection(self):
        calls = []
        for name in ["web_cockpit.py", "robot_vision_client.py"]:
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "CameraStreamThread":
                        calls.append((name, node))
        self.assertEqual(len(calls), 2)
        for name, call in calls:
            with self.subTest(file=name):
                self.assertEqual(call.args, [])
                self.assertEqual(call.keywords, [])


if __name__ == "__main__":
    unittest.main()
