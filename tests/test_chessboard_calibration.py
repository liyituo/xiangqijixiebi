"""棋盘定位和预览接口检查；不会连接摄像头或机械臂。"""

import ast
import math
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, jsonify, request

from vision_calibration import HandEyeMapper


ROOT = Path(__file__).resolve().parents[1]


class FakeCamera:
    def __init__(self):
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def get_latest_frame(self):
        return self.frame


def make_preview_app(mapper, camera, telemetry):
    # 只加载预览函数，避免导入大屏时启动硬件和后台线程。
    tree = ast.parse((ROOT / 'web_cockpit.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'api_preview_position')
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {
        'request': request, 'jsonify': jsonify, 'math': math,
        'cam': camera, 'mapper': mapper, 'state_lock': threading.RLock(),
        'telemetry': telemetry,
    }
    exec(compile(module, 'web_cockpit.py', 'exec'), scope)
    app = Flask(__name__)
    app.add_url_rule('/api/preview_position', view_func=scope['api_preview_position'], methods=['POST'])
    return app


class ChessboardPreviewTest(unittest.TestCase):
    def setUp(self):
        self.mapper = HandEyeMapper(str(ROOT / 'calibration_config.json'))
        self.camera = FakeCamera()
        self.telemetry = {'calib_offset_x': 0.0, 'calib_offset_y': 0.0}
        self.client = make_preview_app(self.mapper, self.camera, self.telemetry).test_client()

    def preview(self, x, y):
        return self.client.post('/api/preview_position', json={'norm_x': x, 'norm_y': y})

    def test_four_user_points_and_grid(self):
        expected = [(139.4, 192.4), (418.9, 147.4), (403.4, -181.3), (105.8, -206.4)]
        self.assertEqual(len(self.mapper.grid_points), 90)
        for point, (x, y) in zip(self.mapper.image_points, expected):
            response = self.preview(float(point[0]), float(point[1]))
            self.assertEqual(response.status_code, 200)
            self.assertEqual((response.json['x_mm'], response.json['y_mm']), (x, y))
            rounded = self.preview(round(float(point[0])), round(float(point[1])))
            self.assertEqual(rounded.status_code, 200)
            self.assertAlmostEqual(rounded.json['x_mm'], x, delta=0.5)
            self.assertAlmostEqual(rounded.json['y_mm'], y, delta=0.5)

    def test_outside_or_invalid_points_are_rejected(self):
        for x, y in [(100, 500), (500, -1), (float('nan'), 500)]:
            self.assertEqual(self.preview(x, y).status_code, 400)

    def test_offsets_match_preview(self):
        before = self.preview(500, 500).json
        self.telemetry.update(calib_offset_x=5.0, calib_offset_y=-5.0)
        after = self.preview(500, 500).json
        self.assertAlmostEqual(after['x_mm'], before['x_mm'] + 5.0, places=1)
        self.assertAlmostEqual(after['y_mm'], before['y_mm'] - 5.0, places=1)

    def test_missing_frame_or_changed_size_is_rejected(self):
        self.camera.frame = None
        self.assertEqual(self.preview(500, 500).status_code, 503)
        self.camera.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(self.preview(500, 500).status_code, 400)

    def test_video_draws_grid_on_a_copy(self):
        tree = ast.parse((ROOT / 'web_cockpit.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'generate_video_stream')
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        self.camera.is_running = True
        scope = {
            'cam': self.camera, 'mapper': self.mapper, 'np': np, 'cv2': cv2,
            'time': time, 'state_lock': threading.RLock(), 'telemetry': self.telemetry,
        }
        exec(compile(module, 'web_cockpit.py', 'exec'), scope)
        chunk = next(scope['generate_video_stream']())
        jpeg = chunk.split(b'\r\n\r\n', 1)[1].removesuffix(b'\r\n')
        rendered = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(rendered[:, :, 1].max()), 100)
        self.assertEqual(int(self.camera.frame.max()), 0)


if __name__ == '__main__':
    unittest.main()
