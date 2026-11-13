"""验证合并后的网页、调度器、驱动适配和相机元数据接口。"""

import importlib
import signal
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from motion_tasks import TaskManager
from robot_vision_client import CameraStreamThread, EnhancedRobotDriver


class MergedRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signal_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with patch.dict('os.environ', {'COCKPIT_OFFLINE_INIT': '1'}), \
                patch('xiangqi_agent.PikafishEngine'), \
                patch('pro450_driver.MyCobotPro450Driver.connect') as connect, \
                patch('camera_device.cv2.VideoCapture') as capture:
            cls.web = importlib.import_module('web_cockpit')
            connect.assert_not_called()
            capture.assert_not_called()

    @classmethod
    def tearDownClass(cls):
        cls.web.xiangqi_agent.close()
        cls.web.driver.close()
        for sig, handler in cls.signal_handlers.items():
            signal.signal(sig, handler)

    def setUp(self):
        self.web.task_manager = TaskManager(
            stop_event_provider=self.web._stop_event_provider,
            on_change=self.web._on_task_change,
        )
        self.addCleanup(self.web.task_manager.shutdown)
        self.web.telemetry.update(is_busy=False, drag_teach=False)
        self.client = self.web.app.test_client()

    def wait_for_task(self):
        deadline = time.monotonic() + 2
        while self.web.task_manager.is_busy and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.web.task_manager.is_busy)
        return self.web.task_manager.last()

    def test_status_and_task_queries(self):
        for path in ['/', '/api/status', '/api/tasks', '/api/video_meta', '/api/xiangqi/status']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_sync_action_uses_task_manager_callback_contract(self):
        with patch.object(self.web.executor, 'jog_cartesian', return_value={'success': True}) as jog, \
                patch.object(self.web, 'refresh_pose'):
            response = self.client.post('/api/jog_cartesian', json={'axis': 'x', 'delta': 1})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        jog.assert_called_once_with('x', 1.0)
        self.assertEqual(self.web.task_manager.last().state, 'succeeded')

    def test_async_scan_reserves_robot_before_accepting_other_actions(self):
        entered = threading.Event()
        release = threading.Event()

        def hold_scan(callback, cancel_token):
            entered.set()
            release.wait(2)
            cancel_token.check()

        with patch.object(self.web.executor, 'scan_workspace', side_effect=hold_scan), \
                patch.object(self.web.executor, 'jog_cartesian') as jog, \
                patch.object(self.web, 'refresh_pose'):
            try:
                response = self.client.post('/api/scan')
                self.assertEqual(response.status_code, 200)
                self.assertIsInstance(response.json['task_id'], int)
                self.assertTrue(entered.wait(1))
                busy = self.client.post('/api/jog_cartesian', json={'axis': 'x', 'delta': 1})
                self.assertEqual(busy.status_code, 409)
                jog.assert_not_called()
            finally:
                release.set()
                record = self.wait_for_task()
        self.assertEqual(record.state, 'succeeded')

    def test_command_failure_is_recorded(self):
        with patch.object(self.web, 'execute_embodied_instruction', return_value={'success': False, 'error': 'test_failure'}):
            response = self.client.post('/api/command', json={'instruction': '回到HOME'})
            self.assertEqual(response.status_code, 200)
            record = self.wait_for_task()
        self.assertEqual(record.state, 'failed')
        self.assertEqual(record.error, 'test_failure')

    def test_camera_metadata_reaches_web_response(self):
        camera = CameraStreamThread.__new__(CameraStreamThread)
        camera.lock = threading.Lock()
        camera.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera.frame_id = 17
        camera.frame_timestamp = time.time()
        camera.is_available = True
        camera.consecutive_failures = 0
        camera.last_error = None
        with patch.object(self.web, 'cam', camera):
            response = self.client.get('/api/video_meta')
            self.assertTrue(response.json['is_fresh'])
            self.assertEqual(response.json['frame_id'], 17)
            self.assertIsInstance(response.json['age_sec'], (int, float))
            camera.frame_timestamp -= 10
            self.assertFalse(self.client.get('/api/video_meta').json['is_fresh'])

    def test_driver_adapter_forwards_feedback_options_and_command_results(self):
        driver = EnhancedRobotDriver(auto_connect=False)
        token = object()
        with patch.object(driver.backend, 'get_coords', return_value=None) as feedback, \
                patch.object(driver.backend, 'send_coords', return_value={'ok': False}) as move:
            self.assertIsNone(driver.get_coords(allow_cache=False))
            self.assertEqual(driver.send_coords([200, 0, 250, 0, 0, 0], cancel_token=token), {'ok': False})
        feedback.assert_called_once_with(allow_cache=False)
        move.assert_called_once_with([200, 0, 250, 0, 0, 0], 15, cancel_token=token)
        driver.close()

    def test_start_result_and_idle_shutdown(self):
        agent = self.web.xiangqi_agent
        agent.stop_game()
        with patch('xiangqi_agent.threading.Thread') as thread:
            thread.return_value.is_alive.return_value = False
            self.assertTrue(agent.start_game())
            agent.stop_game()
        agent.close()
        self.assertIsNone(agent.thread)
        self.assertFalse(agent.is_running)


if __name__ == '__main__':
    unittest.main()
