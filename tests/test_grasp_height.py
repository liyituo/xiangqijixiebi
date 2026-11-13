"""用模拟驱动检查抓取高度；不会连接或控制机械臂。"""

import contextlib
import io
import unittest
from unittest.mock import patch

from vision_tools import VisionActionExecutor


class FakeMapper:
    hover_z = 310.0

    def bbox_to_robot_center(self, bbox):
        return 200.0, -50.0, 189.0


class FakeDriver:
    def __init__(self):
        self.cartesian_commands = []

    def send_coords(self, coords, speed=15):
        self.cartesian_commands.append(list(coords))

    def send_angles(self, angles, speed=15):
        pass

    def set_gripper_value(self, value, speed=50):
        pass


class GraspHeightTest(unittest.TestCase):
    def test_grasp_and_hover_use_configured_table_offsets(self):
        driver = FakeDriver()
        executor = VisionActionExecutor(driver, FakeMapper())
        with patch('vision_tools.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            result = executor.execute_visual_pick_and_place('棋子', [0, 0, 1, 1])

        self.assertTrue(result['success'])
        self.assertEqual([coords[2] for coords in driver.cartesian_commands[:3]],
                         [234.0, 187.0, 234.0])

    def test_manual_z_trim_still_applies(self):
        driver = FakeDriver()
        executor = VisionActionExecutor(driver, FakeMapper())
        with patch('vision_tools.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
            executor.execute_visual_pick_and_place('棋子', [0, 0, 1, 1], offset_trim=(0, 0, 5))
        self.assertEqual(driver.cartesian_commands[1][2], 192.0)


if __name__ == '__main__':
    unittest.main()
