"""离线检查点击抓取的忙碌与失败反馈；不连接机械臂。"""

import ast
import math
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parents[1]


class InlineThread:
    def __init__(self, target, daemon=False):
        self.target = target

    def start(self):
        self.target()


class FakeExecutor:
    def __init__(self):
        self.result = {'success': False, 'error': '目标位置不可达'}
        self.calls = 0
        self.received = []
        self.leds = []

    def set_status_led(self, status):
        self.leds.append(status)

    def execute_visual_pick_and_place(self, *args, **kwargs):
        self.calls += 1
        self.received.append((args, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_app(telemetry, executor, logs, connected=True):
    tree = ast.parse((ROOT / 'web_cockpit.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'api_click_pick')
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {
        'request': request, 'jsonify': jsonify, 'math': math,
        'state_lock': threading.RLock(), 'telemetry': telemetry,
        'driver': SimpleNamespace(is_connected=connected), 'executor': executor,
        'mapper': SimpleNamespace(pixel_to_robot=lambda x, y: (200.0, 0.0, 189.0)),
        'add_log': logs.append, 'refresh_pose': lambda: None,
        'threading': SimpleNamespace(Thread=InlineThread),
    }
    exec(compile(module, 'web_cockpit.py', 'exec'), scope)
    app = Flask(__name__)
    app.add_url_rule('/api/click_pick', view_func=scope['api_click_pick'], methods=['POST'])
    return app


class ClickPickApiTest(unittest.TestCase):
    def setUp(self):
        self.telemetry = {
            'is_busy': False, 'drag_teach': False, 'live_detected_objects': [],
            'calib_offset_x': 0.0, 'calib_offset_y': 0.0, 'calib_offset_z': 0.0,
        }
        self.executor = FakeExecutor()
        self.logs = []
        self.client = make_app(self.telemetry, self.executor, self.logs).test_client()

    def test_busy_request_is_rejected_without_starting_another_task(self):
        self.telemetry['is_busy'] = True
        response = self.client.post('/api/click_pick', json={'norm_x': 500, 'norm_y': 500})
        self.assertFalse(response.json['success'])
        self.assertIn('正在执行', response.json['error'])
        self.assertEqual(self.executor.calls, 0)

    def test_failed_task_does_not_claim_success_and_releases_busy(self):
        self.client.post('/api/click_pick', json={'norm_x': 500, 'norm_y': 500})
        self.assertIn('目标位置不可达', self.telemetry['current_step'])
        self.assertFalse(self.telemetry['is_busy'])
        self.assertIn('ERROR', self.executor.leds)
        self.assertFalse(any('圆满完成' in item for item in self.logs))

    def test_disconnected_robot_is_reported_without_starting_task(self):
        client = make_app(self.telemetry, self.executor, self.logs, connected=False).test_client()
        response = client.post('/api/click_pick', json={'norm_x': 500, 'norm_y': 500})
        self.assertEqual(response.status_code, 503)
        self.assertIn('未连接真机', response.json['error'])
        self.assertFalse(self.telemetry['is_busy'])
        self.assertEqual(self.executor.calls, 0)

    def test_motion_exception_releases_busy_and_reports_error(self):
        self.executor.result = RuntimeError('目标姿态无逆解')
        self.client.post('/api/click_pick', json={'norm_x': 500, 'norm_y': 500})
        self.assertIn('目标姿态无逆解', self.telemetry['current_step'])
        self.assertFalse(self.telemetry['is_busy'])

    def test_click_defaults_to_p5_but_explicit_station_is_preserved(self):
        self.client.post('/api/click_pick', json={'norm_x': 500, 'norm_y': 500})
        self.assertEqual(self.executor.received[-1][0][2], 'P5')
        self.client.post('/api/click_pick', json={'norm_x': 500, 'norm_y': 500, 'target_station': 'P3'})
        self.assertEqual(self.executor.received[-1][0][2], 'P3')


if __name__ == '__main__':
    unittest.main()
