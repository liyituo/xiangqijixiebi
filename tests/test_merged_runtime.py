"""验证合并后的网页、调度器、驱动适配和相机元数据接口。"""

import importlib
import signal
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

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

    def test_endgame_home_uses_task_manager_callback_contract(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        detected = {'success': True, 'fen': self.web.xiangqi_agent.board.get_fen(),
                    'piece_dict': {'e0': 'K'}}
        with patch.object(self.web.executor, 'move_to_station', return_value={'success': True}) as home, \
                patch.object(self.web.cam, 'get_latest_frame_with_meta', return_value=(frame, {'fresh': True})), \
                patch('xiangqi_endgame.detect_endgame_from_image', return_value=detected), \
                patch.object(self.web.xiangqi_agent, 'start_game', return_value=True), \
                patch('web_cockpit.time.sleep'):
            response = self.client.post('/api/xiangqi/endgame_detect')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        record = self.web.task_manager.last()
        self.assertEqual(record.state, 'succeeded')
        home.assert_called_once_with('HOME', cancel_token=record.token)

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

    def _run_vlm_command(self, instruction, model_result, objects=None):
        frame = np.zeros((720, 1280, 3), np.uint8)
        executor = MagicMock()
        executor.execute_visual_pick_and_place.return_value = {'success': True}
        executor.move_to_station.return_value = {'success': True}
        with patch.object(self.web, 'executor', executor), \
                patch.object(self.web.cam, 'get_latest_frame', return_value=frame), \
                patch.object(self.web, 'call_deepseek_embodied_vlm', return_value=model_result) as vlm, \
                patch.object(self.web, 'detect_all_table_objects', return_value=objects or []) as detect, \
                patch.object(self.web, 'refresh_pose'), \
                patch('motion_tasks.CancelToken.wait'):
            response = self.client.post('/api/command', json={'instruction': instruction})
            self.assertEqual(response.status_code, 200)
            record = self.wait_for_task()
        return record, executor, vlm, detect

    def test_vlm_pick_accepts_nested_and_flat_model_results(self):
        bbox = [410, 420, 480, 490]
        results = [
            {'intent': 'pick_and_place', 'has_tool_call': True,
             'arguments': {'target_object': '黄色长尾夹', 'normalized_bbox': bbox, 'target_station': 'P5'}},
            {'intent': 'pick_and_place', 'target': '黄色长尾夹', 'bbox': bbox, 'destination': 'P5'},
        ]
        for result in results:
            with self.subTest(result=result):
                record, executor, vlm, detect = self._run_vlm_command('识别夹子，并且移到P5区域', result)
                self.assertEqual(record.state, 'succeeded', record.error)
                self.assertEqual(record.result['moved_count'], 1)
                self.assertEqual(executor.execute_visual_pick_and_place.call_args.args,
                                 ('黄色长尾夹', bbox, 'P5'))
                vlm.assert_called_once()
                detect.assert_not_called()

    def test_vlm_object_transfers_are_classified_before_station_fallback(self):
        cases = [
            ('把螺丝刀移到P5', 'PICK_AND_PLACE', 'P5'),
            ('将尺子移至P2', 'PICK_AND_PLACE', 'P2'),
            ('把笔移动到P3', 'PICK_AND_PLACE', 'P3'),
            ('识别螺丝刀，并移到P5', 'PICK_AND_PLACE', 'P5'),
            ('把机械臂旁的螺丝刀移到P5', 'PICK_AND_PLACE', 'P5'),
            ('把机械臂移到P5', 'MOVE_TO_STATION', 'P5'),
            ('把机械臂缓慢地移到P5', 'MOVE_TO_STATION', 'P5'),
            ('把夹爪移到P5', 'MOVE_TO_STATION', 'P5'),
            ('移动到P5工位', 'MOVE_TO_STATION', 'P5'),
        ]
        model = {'intent': 'pick_and_place', 'target': '蓝色手柄螺丝刀',
                 'bbox': [300, 450, 390, 610], 'destination': 'P5'}
        for instruction, intent, destination in cases:
            with self.subTest(instruction=instruction):
                parsed = self.web.parse_instruction(instruction, model)
                self.assertEqual(parsed.intent, intent)
                self.assertEqual(parsed.destination, destination)

    def test_vlm_consecutive_requests_pick_each_new_object(self):
        frames = [np.zeros((720, 1280, 3), np.uint8), np.full((720, 1280, 3), 100, np.uint8)]
        targets = [('黄色长尾夹', [110, 120, 200, 210]), ('蓝色手柄螺丝刀', [300, 450, 390, 610])]
        models = [{'intent': 'pick_and_place',
                   'arguments': {'target_object': name, 'normalized_bbox': bbox, 'target_station': 'P5'}}
                  for name, bbox in targets]
        executor = MagicMock()
        executor.execute_visual_pick_and_place.return_value = {'success': True}
        with patch.object(self.web, 'executor', executor), \
                patch.object(self.web.cam, 'get_latest_frame') as camera, \
                patch.object(self.web, 'call_deepseek_embodied_vlm', side_effect=models) as vlm, \
                patch.object(self.web, 'detect_all_table_objects') as detect, \
                patch.object(self.web, 'refresh_pose'), \
                patch('motion_tasks.CancelToken.wait'):
            for instruction, frame in zip(('识别夹子，并且移到P5区域', '把螺丝刀移到P5'), frames):
                camera.return_value = frame
                response = self.client.post('/api/command', json={'instruction': instruction})
                self.assertEqual(response.status_code, 200)
                record = self.wait_for_task()
                self.assertEqual(record.state, 'succeeded', record.error)
                self.assertEqual(record.result.get('moved_count'), 1)

        self.assertEqual([call.args for call in executor.execute_visual_pick_and_place.call_args_list],
                         [(name, bbox, 'P5') for name, bbox in targets])
        self.assertEqual(vlm.call_count, 2)
        self.assertIs(vlm.call_args_list[0].args[1], frames[0])
        self.assertIs(vlm.call_args_list[1].args[1], frames[1])
        executor.move_to_station.assert_not_called()
        detect.assert_not_called()

    def test_vlm_negated_pick_is_rejected_before_model_call(self):
        record, executor, vlm, detect = self._run_vlm_command(
            '不要把夹子移到P5', {'intent': 'pick_and_place'})
        self.assertEqual(record.state, 'failed')
        self.assertEqual(record.result['intent'], 'reject')
        executor.execute_visual_pick_and_place.assert_not_called()
        vlm.assert_not_called()
        detect.assert_not_called()

    def test_vlm_batch_uses_count_and_rejects_invalid_boxes(self):
        objects = [
            {'name': '工件', 'bbox': ['bad', 0, 1, 1]},
            {'name': '工件', 'bbox': [100, 100, 160, 160]},
            {'name': '工件', 'bbox': [500, 500, 560, 560]},
        ]
        model = {'intent': 'pick_and_place', 'target': '工件', 'bbox': ['bad', 0, 1, 1]}
        for instruction in ('把两个工件搬到P5', '把全部工件搬到P5'):
            with self.subTest(instruction=instruction):
                record, executor, _, _ = self._run_vlm_command(instruction, model, objects)
                self.assertEqual(record.state, 'succeeded', record.error)
                self.assertEqual(record.result['moved_count'], 2)
                self.assertEqual([item.args[1] for item in executor.execute_visual_pick_and_place.call_args_list],
                                 [objects[1]['bbox'], objects[2]['bbox']])

    def test_vlm_query_and_scan_use_parser_intents(self):
        model = {'intent': 'chat', 'reply': '桌面上有两个夹子'}
        record, executor, _, _ = self._run_vlm_command('桌面上有什么', model)
        self.assertEqual(record.state, 'succeeded', record.error)
        self.assertEqual(record.result['reply'], model['reply'])
        executor.execute_visual_pick_and_place.assert_not_called()
        record, executor, _, _ = self._run_vlm_command('扫描工作台', {'intent': 'scan'})
        self.assertEqual(record.state, 'succeeded', record.error)
        executor.scan_workspace.assert_called_once()
        executor.execute_visual_pick_and_place.assert_not_called()

    def test_vlm_station_move_uses_requested_destination(self):
        record, executor, _, _ = self._run_vlm_command('移动到P2工位', {'intent': 'move_to_station'})
        self.assertEqual(record.state, 'succeeded', record.error)
        executor.move_to_station.assert_called_once_with('P2', cancel_token=record.token)
        executor.execute_visual_pick_and_place.assert_not_called()

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
