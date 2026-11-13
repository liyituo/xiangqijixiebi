import ast
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask, jsonify, request

import vision_tools
from vision_calibration import HandEyeMapper
from xiangqi_agent import XiangqiAgent, XiangqiGridMapper


WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_SRC = os.path.join(WORKSPACE_DIR, "calibration_config.json")
BOARD_SRC = os.path.join(WORKSPACE_DIR, "xiangqi_board_config.json")
WEB_COCKPIT_PATH = os.path.join(WORKSPACE_DIR, "web_cockpit.py")


class TestHandEyeMapperOffsetPersistence(unittest.TestCase):
    """验证手眼微调偏置在标定文件中的读取、持久化、热重载及隔离契约"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_calib = os.path.join(self.tmpdir.name, "calibration_config.json")
        shutil.copy2(CALIB_SRC, self.tmp_calib)
        with open(self.tmp_calib, "r", encoding="utf-8") as config_file:
            fixture = json.load(config_file)
        fixture["manual_offset_mm"] = [0.0, 0.0, 0.0]
        with open(self.tmp_calib, "w", encoding="utf-8") as config_file:
            json.dump(fixture, config_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_persist_reload_and_apply_flag(self):
        mapper = HandEyeMapper(config_path=self.tmp_calib)
        self.assertEqual(mapper.get_offset(), (0.0, 0.0, 0.0))

        base_xyz = mapper.pixel_to_robot(500.0, 500.0)
        self.assertEqual(mapper.pixel_to_robot(500.0, 500.0, apply_offset=True), base_xyz)

        # 保存微调偏置并持久化到磁盘
        saved = mapper.set_offset(3.5, -2.0, 1.5, persist=True)
        self.assertEqual(saved, (3.5, -2.0, 1.5))

        # 默认 apply_offset=False 保持兼容（不重复叠加 web_cockpit 已有偏置）
        self.assertEqual(mapper.pixel_to_robot(500.0, 500.0), base_xyz)
        self.assertEqual(
            mapper.pixel_to_robot(500.0, 500.0, apply_offset=True),
            (
                round(base_xyz[0] + 3.5, 1),
                round(base_xyz[1] - 2.0, 1),
                round(base_xyz[2] + 1.5, 1),
            ),
        )

        # 验证 JSON 结构完好且新实例可直接加载持久化偏置
        with open(self.tmp_calib, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertEqual(disk_data["manual_offset_mm"], [3.5, -2.0, 1.5])
        self.assertIn("image_points", disk_data)
        self.assertIn("robot_points", disk_data)

        reloaded_mapper = HandEyeMapper(config_path=self.tmp_calib)
        self.assertEqual(reloaded_mapper.get_offset(), (3.5, -2.0, 1.5))

        # 模拟外部修改磁盘文件并触发热重载
        disk_data["manual_offset_mm"] = [-1.2, 4.0, -0.5]
        with open(self.tmp_calib, "w", encoding="utf-8") as f:
            json.dump(disk_data, f, ensure_ascii=False, indent=2)
        future_mtime = time.time() + 5.0
        os.utime(self.tmp_calib, (future_mtime, future_mtime))

        self.assertTrue(mapper.check_reload())
        self.assertEqual(mapper.get_offset(), (-1.2, 4.0, -0.5))


class TestXiangqiGridAndAgentOffsets(unittest.TestCase):
    """验证象棋棋盘坐标映射与抓/放/弃子物理执行器完整应用手眼微调偏置"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_calib = os.path.join(self.tmpdir.name, "calibration_config.json")
        self.tmp_board = os.path.join(self.tmpdir.name, "xiangqi_board_config.json")
        shutil.copy2(CALIB_SRC, self.tmp_calib)
        shutil.copy2(BOARD_SRC, self.tmp_board)

        self.mapper = HandEyeMapper(config_path=self.tmp_calib)
        self.grid_mapper = XiangqiGridMapper(
            hand_eye_mapper=self.mapper,
            config_path=self.tmp_board,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_grid_mapper_applies_explicit_and_persisted_offsets(self):
        base_xyz = self.grid_mapper.get_square_robot_xyz("e4", offset_trim=(0.0, 0.0, 0.0))
        shifted_xyz = self.grid_mapper.get_square_robot_xyz("e4", offset_trim=(2.5, -3.5, 1.0))
        self.assertEqual(
            shifted_xyz,
            (
                round(base_xyz[0] + 2.5, 1),
                round(base_xyz[1] - 3.5, 1),
                round(base_xyz[2] + 1.0, 1),
            ),
        )

        self.mapper.set_offset(-1.5, 2.0, -0.5, persist=True)
        auto_xyz = self.grid_mapper.get_square_robot_xyz("e4")
        self.assertEqual(
            auto_xyz,
            (
                round(base_xyz[0] - 1.5, 1),
                round(base_xyz[1] + 2.0, 1),
                round(base_xyz[2] - 0.5, 1),
            ),
        )

    @staticmethod
    def _build_mock_driver():
        class DummyDriver:
            def __init__(self):
                self.is_connected = True
                self.stop_flag = False
                self.send_coords_history = []
                self.send_angles_history = []
                self.gripper_history = []

            def send_coords(self, coords, speed=20, **kwargs):
                self.send_coords_history.append(list(coords))
                return True

            def send_angles(self, angles, speed=20, **kwargs):
                self.send_angles_history.append(list(angles))
                return True

            def set_gripper_value(self, value, speed=20, **kwargs):
                self.gripper_history.append(value)
                return True

            def set_gripper(self, value, speed=20, **kwargs):
                self.gripper_history.append(value)
                return True

            def set_gripper_state(self, state, speed=20, **kwargs):
                self.gripper_history.append(state)
                return True

            def get_coords(self):
                return [250.0, 0.0, 260.0, -178.5, -3.1, 46.0]

            def get_angles(self):
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            def set_color(self, r, g, b):
                return True

            def is_moving(self):
                return False

        return DummyDriver()

    @staticmethod
    def _extract_pick_and_place_targets(coords):
        # 执行顺序依次为: 抓取悬停 -> 抓取点 -> 抬起 -> 放置悬停 -> 放置点
        # 局部极小 Z 点对应抓取点与放置点
        surface_pts = []
        for idx, pt in enumerate(coords):
            z = pt[2]
            prev_z = coords[idx - 1][2] if idx > 0 else float("inf")
            next_z = coords[idx + 1][2] if idx + 1 < len(coords) else float("inf")
            if z < prev_z and z <= next_z:
                surface_pts.append(pt)
        return surface_pts

    def test_physical_move_and_capture_follow_dynamic_offsets_without_double_count(self):
        driver = self._build_mock_driver()
        executor = vision_tools.VisionActionExecutor(driver, self.mapper)
        current_offset = [0.0, 0.0, 0.0]

        with patch("xiangqi_agent.PikafishEngine", return_value=MagicMock()):
            agent = XiangqiAgent(
                driver=driver,
                executor=executor,
                grid_mapper=self.grid_mapper,
                offset_provider=lambda: tuple(current_offset),
            )

        with patch.object(vision_tools.time, "sleep", return_value=None):
            agent._physical_move_piece("e7", "e5")
            base_surfaces = self._extract_pick_and_place_targets(driver.send_coords_history)
            self.assertEqual(len(base_surfaces), 2)
            base_pick, base_place = base_surfaces
            self.assertAlmostEqual(base_place[2] - base_pick[2], 2.0, places=1)

            # 对弈中途手动调整微调偏置，无需重启智能体即应立即生效
            driver.send_coords_history.clear()
            driver.send_angles_history.clear()
            current_offset[:] = [4.0, -3.0, 2.0]
            agent._physical_move_piece("e7", "e5")
            shifted_surfaces = self._extract_pick_and_place_targets(driver.send_coords_history)
            self.assertEqual(len(shifted_surfaces), 2)
            shifted_pick, shifted_place = shifted_surfaces

            # 拾取点与落子点均严格按 (+4.0, -3.0, +2.0) 平移且无二次叠加
            self.assertAlmostEqual(shifted_pick[0] - base_pick[0], 4.0, places=1)
            self.assertAlmostEqual(shifted_pick[1] - base_pick[1], -3.0, places=1)
            self.assertAlmostEqual(shifted_pick[2] - base_pick[2], 2.0, places=1)

            self.assertAlmostEqual(shifted_place[0] - base_place[0], 4.0, places=1)
            self.assertAlmostEqual(shifted_place[1] - base_place[1], -3.0, places=1)
            self.assertAlmostEqual(shifted_place[2] - base_place[2], 2.0, places=1)
            self.assertAlmostEqual(shifted_place[2] - shifted_pick[2], 2.0, places=1)

            # 吃子丢入弃子盒 P4：仅对棋盘拾取点生效，P4 固定工位不偏移
            driver.send_coords_history.clear()
            driver.send_angles_history.clear()
            agent._physical_pick_and_place_square("e5", dest_station="P4")
            capture_z_vals = [round(c[2], 1) for c in driver.send_coords_history]
            self.assertIn(round(base_pick[2] + 2.0, 1), capture_z_vals)
            self.assertIn(vision_tools.PRESET_JOINTS["P4"], driver.send_angles_history)

    def test_custom_dest_safety_guard_and_agent_error_propagation(self):
        driver = self._build_mock_driver()
        executor = vision_tools.VisionActionExecutor(driver, self.mapper)

        # 自定义终点越界时，在发起任何机械臂移动前必须被拦截
        res = executor.execute_visual_pick_and_place(
            target_name="越界测试棋子",
            bbox=[460, 460, 540, 540],
            target_station=[999.0, 0.0, 184.2],
            speed=25,
            offset_trim=(0.0, 0.0, 0.0),
        )
        self.assertFalse(res.get("success", True))
        self.assertIn("安全拦截", res.get("error", ""))
        self.assertEqual(driver.send_coords_history, [])

        # 若 executor 返回失败，XiangqiAgent 必须抛出 RuntimeError 而非静默吞掉
        failing_executor = MagicMock()
        failing_executor.execute_visual_pick_and_place.return_value = {
            "success": False,
            "error": "模拟限位拦截",
        }
        with patch("xiangqi_agent.PikafishEngine", return_value=MagicMock()):
            agent = XiangqiAgent(
                driver=driver,
                executor=failing_executor,
                grid_mapper=self.grid_mapper,
                offset_provider=lambda: (1.0, 2.0, 3.0),
            )
        with self.assertRaises(RuntimeError):
            agent._physical_move_piece("e7", "e5")


class TestWebCockpitOffsetWiring(unittest.TestCase):
    """通过 AST 验证 web_cockpit.py 的偏置接口、持久化同步、象棋状态路由与前端展示"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_calib = os.path.join(self.tmpdir.name, "calibration_config.json")
        shutil.copy2(CALIB_SRC, self.tmp_calib)
        with open(WEB_COCKPIT_PATH, "r", encoding="utf-8") as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _compile_functions(self, names, namespace):
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                mod = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(mod)
                exec(compile(mod, WEB_COCKPIT_PATH, "exec"), namespace)

    def test_cockpit_api_calib_offset_and_xiangqi_status_sync(self):
        mapper = HandEyeMapper(config_path=self.tmp_calib)
        agent_mock = MagicMock()
        agent_mock.lock = threading.RLock()
        agent_mock.is_running = False
        agent_mock.current_state = "IDLE"
        agent_mock.status_text = "就绪"
        agent_mock.board.turn = "w"
        agent_mock.board.get_fen.return_value = "startpos"
        agent_mock.last_red_move = "--"
        agent_mock.last_black_move = "--"
        agent_mock.move_history = []
        agent_mock.fault_info = None
        agent_mock.game_id = 0

        grid_mock = MagicMock()
        grid_mock.corners_norm = {"a0": [100, 100], "i0": [900, 100], "i9": [900, 900], "a9": [100, 900]}

        ns = {
            "app": Flask(__name__),
            "request": request,
            "jsonify": jsonify,
            "state_lock": threading.RLock(),
            "mapper": mapper,
            "telemetry": {
                "calib_offset_x": 0.0,
                "calib_offset_y": 0.0,
                "calib_offset_z": 0.0,
                "show_xiangqi_grid": True,
            },
            "add_log": lambda msg: None,
            "refresh_pose": lambda: None,
            "sanitize_for_json": lambda x: x,
            "xiangqi_agent": agent_mock,
            "xiangqi_grid_mapper": grid_mock,
        }
        self._compile_functions(
            {
                "_sync_calib_offset_locked",
                "get_calib_offset",
                "api_calib_offset",
                "api_status",
                "api_xiangqi_status",
            },
            ns,
        )

        client = ns["app"].test_client()
        resp = client.post("/api/calib_offset", json={"axis": "x", "delta": 2.5})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["calib_offset_x"], 2.5)

        resp_z = client.post("/api/calib_offset", json={"axis": "z", "delta": -1.0})
        self.assertEqual(resp_z.status_code, 200)

        # 验证写入磁盘并能通过 get_calib_offset 和 /api/xiangqi/status 读取
        self.assertEqual(ns["get_calib_offset"](), (2.5, 0.0, -1.0))
        self.assertEqual(HandEyeMapper(config_path=self.tmp_calib).get_offset(), (2.5, 0.0, -1.0))

        xq_status = client.get("/api/xiangqi/status").get_json()
        self.assertEqual(xq_status["calib_offset"], {"x": 2.5, "y": 0.0, "z": -1.0})

        # 清零同步验证
        client.post("/api/calib_offset", json={"action": "reset"})
        self.assertEqual(ns["get_calib_offset"](), (0.0, 0.0, 0.0))
        self.assertEqual(HandEyeMapper(config_path=self.tmp_calib).get_offset(), (0.0, 0.0, 0.0))

    def test_web_cockpit_wires_offset_provider_and_ui_readout(self):
        self.assertIn("offset_provider=get_calib_offset", self.source)
        self.assertIn('id="xqOffsetSummary"', self.source)
        self.assertIn("d.calib_offset", self.source)


if __name__ == "__main__":
    unittest.main()
