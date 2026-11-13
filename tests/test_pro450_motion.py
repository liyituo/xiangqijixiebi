"""离线验证运动失败、中止与停止；不连接机械臂。"""

import contextlib
import io
import threading
import unittest
from unittest.mock import patch

from pro450_driver import MyCobotPro450Driver
from vision_tools import VisionActionExecutor


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeClient:
    def __init__(self):
        self.commands = []
        self.stop_count = 0
        self.solution = [0.0] * 6
        self.arrived = 0
        self.moving = 0

    def get_angles(self):
        return [0.0] * 6

    def solve_inv_kinematics(self, coords, angles):
        return self.solution

    def send_coords(self, coords, speed, _async=False):
        self.commands.append(('coords', list(coords), _async))
        return 1

    def send_angles(self, angles, speed, _async=False):
        self.commands.append(('angles', list(angles), _async))
        return 1

    def is_in_position(self, target, mode):
        return self.arrived

    def is_moving(self):
        return self.moving

    def stop(self):
        self.stop_count += 1
        return 1


def make_driver(client):
    driver = MyCobotPro450Driver.__new__(MyCobotPro450Driver)
    driver.client = client
    driver.driver_type = 'Pro450Client'
    driver.is_connected = True
    driver.is_free_mode = False
    driver.lock = threading.RLock()
    driver.stop_requested = threading.Event()
    driver.virtual_coords = [0.0] * 6
    driver.virtual_angles = [0.0] * 6
    return driver


class MotionTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.driver = make_driver(self.client)
        self.clock = FakeClock()
        self.target = [200, 0, 250, -178, 0, 45]

    def test_unreachable_pose_is_not_sent(self):
        self.client.solution = 'No solution for conversion'
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, '逆解'):
            self.driver.send_coords(self.target)
        self.assertEqual(self.client.commands, [])

    def test_arrival_uses_async_command_and_updates_pose(self):
        self.client.arrived = 1
        self.driver.send_coords(self.target)
        self.assertEqual(self.client.commands, [('coords', self.target, True)])
        self.assertEqual(self.driver.virtual_coords, self.target)
        self.assertEqual(self.client.stop_count, 0)

    def test_stopped_before_arrival_is_reported_and_stopped(self):
        with patch('pro450_driver.time.monotonic', self.clock.monotonic), \
             patch('pro450_driver.time.sleep', self.clock.sleep), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, '未到达目标'):
            self.driver.send_coords(self.target)
        self.assertLess(self.clock.now, 6.0)
        self.assertEqual(self.client.stop_count, 1)
        self.assertEqual(self.driver.virtual_coords, [0.0] * 6)

    def test_moving_timeout_sends_stop(self):
        self.client.moving = 1
        with patch('pro450_driver.time.monotonic', self.clock.monotonic), \
             patch('pro450_driver.time.sleep', self.clock.sleep), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, '超时'):
            self.driver.send_angles([0, 10, 10, 0, 0, 0])
        self.assertEqual(self.client.stop_count, 1)

    def test_unconfirmed_stop_is_reported(self):
        self.client.stop = lambda: -1
        with patch('pro450_driver.time.monotonic', self.clock.monotonic), \
             patch('pro450_driver.time.sleep', self.clock.sleep), \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, '停止请求未获确认'):
            self.driver.send_coords(self.target)

    def test_emergency_stop_does_not_wait_for_motion_lock(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def slow_status(target, mode):
            entered.set()
            release.wait(2.0)
            return 0

        self.client.is_in_position = slow_status
        self.client.moving = 1

        def move():
            try:
                self.driver.send_coords(self.target)
            except Exception as error:
                errors.append(str(error))

        with contextlib.redirect_stdout(io.StringIO()):
            worker = threading.Thread(target=move)
            worker.start()
            self.assertTrue(entered.wait(1.0))
            stopper = threading.Thread(target=self.driver.emergency_stop)
            stopper.start()
            stopper.join(1.0)
            self.assertFalse(stopper.is_alive(), '停止请求不应等待运动锁')
            release.set()
            worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.client.stop_count, 1)
        self.assertTrue(any('停止' in error for error in errors))

    def test_failed_move_does_not_continue_to_grasp(self):
        class Mapper:
            hover_z = 310.0
            def bbox_to_robot_center(self, bbox):
                return 200.0, -50.0, 189.0

        class Driver:
            def __init__(self): self.actions = []
            def set_gripper_value(self, value, speed=50): self.actions.append(('gripper', value))
            def send_coords(self, coords, speed=15):
                self.actions.append(('move', coords[2]))
                raise RuntimeError('目标位置和姿态无有效逆解')
            def send_angles(self, angles, speed=15): self.actions.append(('angles', angles))

        fake = Driver()
        with patch('vision_tools.time.sleep'), contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, '逆解'):
            VisionActionExecutor(fake, Mapper()).execute_visual_pick_and_place('棋子', [0, 0, 1, 1])
        self.assertEqual(fake.actions, [('gripper', 15), ('move', 234.0)])


if __name__ == '__main__':
    unittest.main()
