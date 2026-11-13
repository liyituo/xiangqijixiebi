# -*- coding: utf-8 -*-
"""
中国象棋具身智能自主对弈中枢 (Embodied Xiangqi Autonomous Playing Agent)
严格遵循 5 步闭环对弈流水线：
1. [拍基准图]: 机械臂停在不遮挡视野的待机位 (NORMAL/HOME)，截取当前干净棋盘基准图 base_frame
2. [轻量监控]: 画面稳定后逐格检查占位变化，等待人手撤离与棋子落稳
3. [棋子追踪 + 规则裁判]: 核验源格清空、目标占位与字形外观，连续帧一致后提交合法红方着法
4. [Pikafish 决策]: 0.5 秒内获取高智力黑方最佳应对着法
5. [机械臂物理执行]: 吃子自动拾取丢入弃子盒，落子平稳转移，动作完毕复位待机并触发下一步基准采样
"""

import os
import sys
import time
import json
import math
import contextlib
import threading
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Callable

from json_store import read_json, atomic_write_json
from xiangqi_engine import XiangqiBoard, PikafishEngine, square_to_coord, coord_to_square, INITIAL_FEN, PIECE_NAMES_CN
from xiangqi_vision import XiangqiMoveDetector, MoveConfirmation
from vision_calibration import HandEyeMapper
from vision_tools import VisionActionExecutor, PRESET_JOINTS, PRESET_STATIONS

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xiangqi_board_config.json")
CHESS_GRIPPER_OPEN_PERCENT = 17

# 默认象棋棋盘在相机画面中的 4 个角点像素坐标 (a0: 红左下, i0: 红右下, i9: 黑右上, a9: 黑左上)
# 对应归一化坐标 0~1000
DEFAULT_BOARD_CORNERS_NORM = {
    "a0": [250.0, 750.0], # 红方九号位车 (左下)
    "i0": [750.0, 750.0], # 红方一号位车 (右下)
    "i9": [750.0, 250.0], # 黑方一号位车 (右上)
    "a9": [250.0, 250.0], # 黑方九号位车 (左上)
}


class XiangqiGridMapper:
    """棋盘 9x10 交叉点与图像像素/机械臂坐标的几何映射器 (带单应性/90点矩阵缓存与版本失效)"""

    def __init__(self, config_path: str = CONFIG_FILE, hand_eye_mapper: Optional[HandEyeMapper] = None):
        self.config_path = config_path
        self.hand_eye_mapper = hand_eye_mapper or HandEyeMapper()
        self._lock = threading.RLock()
        self.corners_norm = dict(DEFAULT_BOARD_CORNERS_NORM)
        self._grid_homography: Optional[np.ndarray] = None
        self._cached_norm_positions: Dict[str, Tuple[float, float]] = {}
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self.load_config()

    def _invalidate_cache(self):
        self._grid_homography = None
        self._cached_norm_positions = {}
        self._cache_key = None

    def _current_cache_key(self) -> Tuple[Any, ...]:
        he_ver = getattr(self.hand_eye_mapper, "version", 0)
        corners_sig = tuple(
            (k, round(float(self.corners_norm[k][0]), 4), round(float(self.corners_norm[k][1]), 4))
            for k in ("a0", "i0", "i9", "a9")
        )
        return (corners_sig, he_ver)

    def _ensure_cache(self):
        with self._lock:
            if hasattr(self.hand_eye_mapper, "check_reload"):
                self.hand_eye_mapper.check_reload()
            key = self._current_cache_key()
            if self._cache_key == key and self._cached_norm_positions:
                return
            src_corners = np.array(
                [[0.0, 0.0], [8.0, 0.0], [8.0, 9.0], [0.0, 9.0]],
                dtype=np.float32,
            )
            dst_corners = np.array(
                [
                    self.corners_norm["a0"],
                    self.corners_norm["i0"],
                    self.corners_norm["i9"],
                    self.corners_norm["a9"],
                ],
                dtype=np.float32,
            )
            h_mat, _ = cv2.findHomography(src_corners, dst_corners)
            if h_mat is None or not np.all(np.isfinite(h_mat)):
                raise RuntimeError("棋盘四角点共线或退化，无法计算单应性矩阵")
            grid_coords = np.array(
                [[[float(c), float(r)] for r in range(10) for c in range(9)]],
                dtype=np.float32,
            )
            transformed = cv2.perspectiveTransform(grid_coords, h_mat)[0]
            cache: Dict[str, Tuple[float, float]] = {}
            idx = 0
            for r in range(10):
                for c in range(9):
                    sq = coord_to_square(c, r)
                    cache[sq] = (float(transformed[idx][0]), float(transformed[idx][1]))
                    idx += 1
            self._grid_homography = h_mat
            self._cached_norm_positions = cache
            self._cache_key = key

    def load_config(self):
        with self._lock:
            if os.path.exists(self.config_path):
                data, err = read_json(self.config_path)
                if err is not None:
                    print(f"⚠️ 读取棋盘配置失败: {err}")
                elif isinstance(data, dict) and "corners_norm" in data:
                    cn = data["corners_norm"]
                    if all(k in cn and len(cn[k]) >= 2 for k in ("a0", "i0", "i9", "a9")):
                        self.corners_norm = {
                            k: [float(cn[k][0]), float(cn[k][1])] for k in ("a0", "i0", "i9", "a9")
                        }
            self._invalidate_cache()

    def save_config(self):
        with self._lock:
            data = {
                "corners_norm": self.corners_norm,
                "description": "中国象棋棋盘 4 角点归一化坐标 (a0:红底左, i0:红底右, i9:黑底右, a9:黑底左)"
            }
            try:
                atomic_write_json(self.config_path, data, indent=2)
            except Exception as e:
                print(f"⚠️ 保存棋盘配置失败: {e}")

    def set_corners(self, a0: List[float], i0: List[float], i9: List[float], a9: List[float]):
        """校验并原子设置 4 个外角点 (0~1000 归一化坐标)，同时刷新 90 点缓存。"""
        raw = {"a0": a0, "i0": i0, "i9": i9, "a9": a9}
        parsed: Dict[str, List[float]] = {}
        for k, pt in raw.items():
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise ValueError(f"角点 {k} 必须为长度为 2 的坐标")
            x, y = float(pt[0]), float(pt[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"角点 {k} 包含非有限数值: {pt!r}")
            if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
                raise ValueError(f"角点 {k} 超出 [0, 1000] 画面范围: {pt!r}")
            parsed[k] = [x, y]

        pts = np.array([parsed["a0"], parsed["i0"], parsed["i9"], parsed["a9"]], dtype=np.float32)
        area = float(cv2.contourArea(pts))
        if area < 500.0:
            raise ValueError(f"棋盘四角点面积过小或共线 (area={area:.1f})")

        with self._lock:
            self.corners_norm.update(parsed)
            self._invalidate_cache()
            self._ensure_cache()
            self.save_config()

    def get_square_norm_pos(self, sq: str) -> Tuple[float, float]:
        """从缓存读取方格 (如 'e2') 的归一化坐标 (norm_x, norm_y)；必要时按版本重建。"""
        square_to_coord(sq)
        sq_key = sq.strip().lower()
        self._ensure_cache()
        return self._cached_norm_positions[sq_key]

    def get_square_pixel_pos(self, sq: str, img_w: int, img_h: int) -> Tuple[int, int]:
        nx, ny = self.get_square_norm_pos(sq)
        return int(nx * img_w / 1000.0), int(ny * img_h / 1000.0)

    def get_square_robot_xyz(
        self,
        sq: str,
        offset_trim: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        """获取方格对应的真实物理机械臂坐标 (X, Y, Z)，支持叠加手眼微调偏置"""
        nx, ny = self.get_square_norm_pos(sq)
        x_mm, y_mm, z_mm = self.hand_eye_mapper.pixel_to_robot(nx, ny)

        if offset_trim is None:
            offset_trim = (
                self.hand_eye_mapper.get_offset()
                if hasattr(self.hand_eye_mapper, "get_offset")
                else (0.0, 0.0, 0.0)
            )

        ox, oy, oz = offset_trim
        return (
            round(x_mm + float(ox), 1),
            round(y_mm + float(oy), 1),
            round(z_mm + float(oz), 1),
        )

    def get_board_bounding_rect_px(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """获取整个棋盘在像素空间中的包围盒 [min_x, min_y, max_x, max_y]"""
        xs = [self.corners_norm[k][0] * img_w / 1000.0 for k in ["a0", "i0", "i9", "a9"]]
        ys = [self.corners_norm[k][1] * img_h / 1000.0 for k in ["a0", "i0", "i9", "a9"]]
        pad = 20
        return (
            max(0, int(min(xs) - pad)),
            max(0, int(min(ys) - pad)),
            min(img_w, int(max(xs) + pad)),
            min(img_h, int(max(ys) + pad))
        )


class XiangqiAgent:
    """中国象棋自主对弈智能体闭环管理器 (支持共享运动锁、分段执行与 FAULT 状态机、光照/遮挡校验)"""

    def __init__(
        self,
        driver,
        executor: VisionActionExecutor,
        grid_mapper: XiangqiGridMapper,
        offset_provider: Optional[Callable[[], Tuple[float, float, float]]] = None,
        motion_session: Optional[Callable[[str, str], Any]] = None,
        frame_provider: Optional[Callable[[], Tuple[Optional[np.ndarray], Dict[str, Any]]]] = None,
        pikafish: Optional[Any] = None,
    ):
        self.driver = driver
        self.executor = executor
        self.grid_mapper = grid_mapper
        self.offset_provider = offset_provider
        self.motion_session = motion_session
        self.frame_provider = frame_provider
        self.board = XiangqiBoard()
        self.pikafish = pikafish if pikafish is not None else PikafishEngine()

        # 残局与视角旋转支持 (当物理棋盘红方在顶部、黑方在底部时，is_rotated=True)
        self.is_rotated: bool = False
        self.is_endgame_mode: bool = False

        # 对弈闭环状态机与并发控制
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.lock = threading.RLock()
        self.game_id = 0
        self.board_version = 0

        # 视觉监控状态
        self.base_frame: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.stable_duration: float = 0.0
        self.last_check_time: float = time.time()
        self.last_detection_info: Dict[str, Any] = {}
        self.move_detector = XiangqiMoveDetector()
        self.move_confirmation = MoveConfirmation()
        self._camera_sample_key = None
        self._baseline_home_ready = False
        self._baseline_ignored_squares = ()
        self._turn_snapshot = None
        self._illegal_move = None

        # 对弈信息记录 (IDLE, WAIT_BASE, MONITOR, VERIFY, VERIFY_POSITION, WAIT_UNDO, PIKAFISH, ARM_MOVE, FAULT)
        self.current_state = "IDLE"
        self.status_text = "待机中"
        self.last_error: Optional[str] = None
        self.fault_info: Optional[Dict[str, Any]] = None
        self.move_history: List[Dict[str, Any]] = []
        self.last_red_move: Optional[str] = None
        self.last_black_move: Optional[str] = None
        self._last_black_std_move: Optional[str] = None

    def _motion_ctx(self, task_type: str, description: str):
        if callable(self.motion_session):
            return self.motion_session(task_type, description)
        return contextlib.nullcontext(None)

    def start_game(self, fen: str = INITIAL_FEN, is_rotated: bool = False, is_endgame: bool = False):
        """校验 FEN 后开始或重置新对局（递增 game_id 防止旧对弈线程写脏）。"""
        ok_fen, fen_err = XiangqiBoard.validate_fen(fen)
        if not ok_fen:
            raise ValueError(f"无法启动对局，FEN 非法: {fen_err}")

        # 若旧对弈线程仍在运行，先停止并等待退出
        if self.thread and self.thread.is_alive():
            self.stop_game(timeout=3.0)

        with self.lock:
            self.board.set_fen(fen)
            self.game_id += 1
            self.board_version += 1
            self.is_rotated = is_rotated
            self.is_endgame_mode = is_endgame
            self.move_history.clear()
            self.last_red_move = None
            self.last_black_move = None
            self._last_black_std_move = None
            self.last_error = None
            self.fault_info = None
            self.base_frame = None
            self._baseline_ignored_squares = ()
            self._turn_snapshot = None
            self._illegal_move = None
            self.prev_frame = None
            self.stable_duration = 0.0
            self.last_detection_info = {}
            self.move_detector.reset()
            self.move_confirmation.reset()
            self._baseline_home_ready = False
            self._stop_event.clear()
            self.is_running = True
            self.current_state = "WAIT_BASE"
            self.status_text = f"启动对弈 ({'残局模式' if is_endgame else '标准初始局'})：复位至待机位，拍摄初始基准图..."
            gid = self.game_id

        self.thread = threading.Thread(
            target=self._battle_loop,
            args=(gid,),
            daemon=True,
            name=f"XiangqiLoop-{gid}",
        )
        self.thread.start()

        return True

    def stop_game(self, timeout: float = 4.0):
        """请求停止对弈并有界等待后台线程退出。"""
        with self.lock:
            self.is_running = False
            self._stop_event.set()
            if self.current_state != "FAULT":
                self.current_state = "IDLE"
                self.status_text = "对弈已停止"
            worker = self.thread
        if worker and worker.is_alive() and threading.current_thread() is not worker:
            worker.join(timeout=timeout)
        with self.lock:
            if self.thread is worker and (worker is None or not worker.is_alive()):
                self.thread = None

    def clear_fault(self, resume_mode: str = "wait_base"):
        """从 FAULT 故障态显式复位；可选重拍基准继续或恢复到指定阶段。"""
        with self.lock:
            if self.current_state != "FAULT":
                return
            self.last_error = None
            self.fault_info = None
            if resume_mode == "idle":
                self.is_running = False
                self.current_state = "IDLE"
                self.status_text = "故障已清除，当前待机"
                return
            self._stop_event.clear()
            self.move_detector.reset()
            self.move_confirmation.reset()
            self._baseline_home_ready = False
            self.is_running = True
            self.base_frame = None
            self._baseline_ignored_squares = ()
            self._turn_snapshot = None
            self._illegal_move = None
            self.current_state = "WAIT_BASE"
            self.status_text = "故障已清除，正在重拍基准图确认物理棋盘..."
            gid = self.game_id
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._battle_loop, args=(gid,), daemon=True)
            self.thread.start()

    def close(self):
        """释放对弈线程与 Pikafish 子进程资源。"""
        self.stop_game(timeout=3.0)
        if self.pikafish and hasattr(self.pikafish, "close"):
            try:
                self.pikafish.close()
            except Exception:
                pass

    def _sleep_cancelable(self, seconds: float, step: float = 0.05) -> bool:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if self._stop_event.is_set() or not self.is_running:
                return False
            time.sleep(min(step, max(0.0, deadline - time.monotonic())))
        return self.is_running and not self._stop_event.is_set()

    def _enter_fault(self, phase: str, error: Exception, extra: Optional[Dict[str, Any]] = None):
        """进入显式 FAULT 故障态，保留阶段上下文并阻止吞错继续乱走。"""
        err_msg = str(error)
        with self.lock:
            self.is_running = False
            self.current_state = "FAULT"
            self.last_error = err_msg
            self.fault_info = {
                "phase": phase,
                "error": err_msg,
                "board_version": self.board_version,
                "game_id": self.game_id,
                "time": time.strftime("%H:%M:%S"),
                **(extra or {}),
            }
            self.status_text = f"象棋执行故障 [{phase}]: {err_msg}"
        try:
            if self.driver and hasattr(self.driver, "set_color"):
                self.driver.set_color(255, 0, 0)
        except Exception:
            pass
        print(f"❌ [象棋智能体进入 FAULT 态 | phase={phase}]: {err_msg}")

    def _execute_arm_move_phases(self, expected_game_id: int, expected_board_ver: int):
        """
        分段物理执行黑方走子：
        Phase 1 (capture_discard): 若目标格有红子，先抓入弃子盒 P5 并记录 capture_cleared；
        Phase 2 (piece_move): 搬运黑子 from -> to；
        Phase 3 (commit_board): 核对 game_id/board_version 一致后提交走子到棋盘状态。
        """
        phys_move = self.last_black_move
        std_move = self._last_black_std_move or phys_move
        if not phys_move or len(phys_move) != 4:
            raise RuntimeError(f"黑方待执行着法无效: {phys_move!r}")

        b_from, b_to = phys_move[:2], phys_move[2:]
        std_to = std_move[2:]
        target_piece = self.board.get_piece(std_to)
        needs_capture = target_piece != "." and target_piece.isupper()
        capture_cleared = False

        with self._motion_ctx("xiangqi_move", f"象棋执黑走子 {b_from}->{b_to}") as cancel_token:
            try:
                if needs_capture:
                    self.status_text = f"阶段1/2 [吃子清理]: 移除目标格 {b_to} 红方棋子至 P5..."
                    self._physical_pick_and_place_square(b_to, dest_station="P5", cancel_token=cancel_token)
                    capture_cleared = True
                    if not self._sleep_cancelable(0.3):
                        raise RuntimeError("对弈已停止")

                self.status_text = f"阶段2/2 [黑方落子]: 搬运黑子 {b_from} -> {b_to}..."
                self._physical_move_piece(b_from, b_to, cancel_token=cancel_token)
            except Exception as exec_err:
                raise RuntimeError(
                    f"phase={'piece_move' if capture_cleared or not needs_capture else 'capture_discard'} "
                    f"(capture_cleared={capture_cleared}): {exec_err}"
                ) from exec_err

        with self.lock:
            if not self.is_running or self.game_id != expected_game_id or self.board_version != expected_board_ver:
                raise RuntimeError("对局状态已在执行期间被重置或切换，放弃提交走子")
            ok, _ = self.board.apply_move(std_move)
            if not ok:
                raise RuntimeError(f"执行后状态提交失败: {std_move} 不再合法")
            self.board_version += 1
            self._baseline_ignored_squares = (b_from, b_to)
            self._turn_snapshot = {
                "fen": self.board.get_fen(), "frame": None,
                "rotated": self.is_rotated, "board_version": self.board_version,
            }
            self.move_history.append({
                "side": "black",
                "move": phys_move,
                "std_move": std_move,
                "captured": target_piece if needs_capture else None,
                "fen": self.board.get_fen(),
                "time": time.strftime("%H:%M:%S")
            })

    def _battle_loop(self, loop_game_id: int):
        """核心 5 步循环主线程"""
        print(f"🚀 [象棋具身智能体]: 启动人机对弈闭环主循环 (game_id={loop_game_id})！")
        while self.is_running and not self._stop_event.is_set() and self.game_id == loop_game_id:
            try:
                state = self.current_state
                if state == "WAIT_BASE":
                    if self._turn_snapshot is None or self._turn_snapshot["board_version"] != self.board_version:
                        self._turn_snapshot = {
                            "fen": self.board.get_fen(), "frame": None,
                            "rotated": self.is_rotated, "board_version": self.board_version,
                        }
                    if not self._baseline_home_ready:
                        self.status_text = "正在平稳复位至待机位 (NORMAL)，拍摄无遮挡基准图..."
                        try:
                            with self._motion_ctx("xiangqi_reset", "象棋对弈复位拍基准图") as cancel_token:
                                kwargs = {"speed": 25}
                                if cancel_token is not None:
                                    kwargs["cancel_token"] = cancel_token
                                self.driver.send_angles(PRESET_JOINTS["NORMAL"], **kwargs)
                        except Exception as e:
                            self._enter_fault("WAIT_BASE", e)
                            break
                        self._baseline_home_ready = True
                        if not self._sleep_cancelable(0.3):
                            break

                    frame = self._get_fresh_camera_frame()
                    if frame is not None:
                        candidate_base = frame.copy()
                        report = self.move_detector.prepare_base(
                            candidate_base, self.board, self.grid_mapper, self.is_rotated
                        )
                        self.last_detection_info = report
                        if not report['ready']:
                            move, info = self.move_detector.detect_from_position(
                                candidate_base, self.board, self.grid_mapper, self.is_rotated,
                            )
                            if move or info.get("illegal_move"):
                                self.last_detection_info = info
                                self.move_confirmation.reset()
                                self.current_state = "VERIFY_POSITION"
                                self.status_text = "当前占位已匹配红方着法，正在连续帧确认吃子局面..."
                                continue
                            if self.base_frame is not None and self._baseline_ignored_squares and report['missing_red_squares']:
                                move, info = self.move_detector.detect(
                                    self.base_frame, candidate_base, self.board, self.grid_mapper,
                                    self.is_rotated, self._baseline_ignored_squares,
                                )
                                if move or info.get("illegal_move"):
                                    self.last_detection_info = info
                                    self.prev_frame = frame.copy()
                                    self.move_confirmation.reset()
                                    self.current_state = "VERIFY"
                                    self.status_text = "检测到基准图采集期间的红方走子，正在连续帧确认..."
                                    continue
                            self.status_text = report['reason']
                            self._sleep_cancelable(0.4)
                            continue
                        with self.lock:
                            if self.game_id != loop_game_id or not self.is_running:
                                break
                            self.base_frame = candidate_base
                            self._turn_snapshot["frame"] = candidate_base
                            self._baseline_ignored_squares = ()
                            self.prev_frame = frame.copy()
                            self.stable_duration = 0.0
                            self.last_check_time = time.time()
                            self.move_confirmation.reset()
                            self.current_state = "MONITOR"
                            self.status_text = "基准图采集完毕，等待红方走棋并核验棋子离位与落位..."
                    else:
                        self._sleep_cancelable(0.4)
                    continue

                elif state == "WAIT_UNDO":
                    frame = self._get_fresh_camera_frame()
                    if frame is None:
                        self._sleep_cancelable(0.2)
                        continue
                    snapshot = self._turn_snapshot
                    restored, info = self.move_detector.matches_snapshot(
                        snapshot["frame"], frame, XiangqiBoard(snapshot["fen"]), self.grid_mapper,
                        snapshot["rotated"], self._illegal_move.get("illegal_squares", ()),
                    )
                    self.last_detection_info = dict(info, undo_required=True, illegal_move=self._illegal_move["illegal_move"])
                    confirmed = self.move_confirmation.observe(
                        ("undo", snapshot["fen"]) if restored else None, self._camera_sample_key,
                    )
                    self.status_text = f"走子错误，请恢复上一回合棋局。{info['reason']}"
                    if confirmed:
                        with self.lock:
                            if self.game_id != loop_game_id or self.board_version != snapshot["board_version"] or not self.is_running:
                                break
                            self.base_frame = frame.copy()
                            self.prev_frame = frame.copy()
                            snapshot["frame"] = self.base_frame
                            self._baseline_ignored_squares = ()
                            self._illegal_move = None
                            self.move_detector.reset()
                            self.move_confirmation.reset()
                            self.stable_duration = 0.0
                            self.last_check_time = time.time()
                            self.current_state = "MONITOR"
                            self.last_detection_info.update(undo_required=False)
                            self.status_text = "已恢复上一回合棋局，请重新走棋"
                    self._sleep_cancelable(0.2)
                    continue

                elif state == "MONITOR":
                    now = time.time()
                    dt = min(0.5, max(0.0, now - self.last_check_time))
                    self.last_check_time = now

                    curr_frame = self._get_fresh_camera_frame()
                    if curr_frame is None or self.prev_frame is None or self.base_frame is None:
                        self._sleep_cancelable(0.2)
                        continue

                    h, w = curr_frame.shape[:2]
                    bx1, by1, bx2, by2 = self.grid_mapper.get_board_bounding_rect_px(w, h)
                    if bx2 <= bx1 or by2 <= by1:
                        self._sleep_cancelable(0.2)
                        continue
                    roi_curr = curr_frame[by1:by2, bx1:bx2]
                    roi_prev = self.prev_frame[by1:by2, bx1:bx2]

                    frame_diff = cv2.absdiff(roi_curr, roi_prev)
                    motion_score = float(np.mean(frame_diff))

                    if motion_score < 3.2:
                        self.stable_duration += dt
                    else:
                        self.stable_duration = 0.0

                    self.prev_frame = curr_frame.copy()

                    if self.stable_duration >= 1.2:
                        self.current_state = "VERIFY"
                        self.status_text = "画面已稳定，正在核验各格棋子离位和落位..."
                    else:
                        self._sleep_cancelable(0.15)
                    continue

                elif state in ("VERIFY", "VERIFY_POSITION"):
                    curr_frame = self._get_fresh_camera_frame()
                    if curr_frame is None or (state == "VERIFY" and self.base_frame is None):
                        self.move_confirmation.reset()
                        self._sleep_cancelable(0.4)
                        continue

                    reference = None if state == "VERIFY_POSITION" else self.base_frame
                    detected_move = self._detect_moved_square_pair(reference, curr_frame)
                    if not detected_move:
                        illegal = self.last_detection_info.get("illegal_move")
                        if illegal:
                            confirmed = self.move_confirmation.observe(("illegal", illegal), self._camera_sample_key)
                            self.status_text = f"正在核实走子错误 [{illegal}]，连续帧确认 {self.move_confirmation.count}/3..."
                            if confirmed:
                                if self._turn_snapshot is None:
                                    snapshot_frame = self.base_frame if state == "VERIFY" and not self._baseline_ignored_squares else None
                                    self._turn_snapshot = {
                                        "fen": self.board.get_fen(), "frame": snapshot_frame,
                                        "rotated": self.is_rotated, "board_version": self.board_version,
                                    }
                                self._illegal_move = dict(self.last_detection_info)
                                self.move_confirmation.reset()
                                self.current_state = "WAIT_UNDO"
                                self.status_text = self.last_detection_info["reason"]
                            self._sleep_cancelable(0.2)
                            continue
                        self.move_confirmation.reset()
                        reason = self.last_detection_info.get("reason", "未锁定高置信度着法")
                        self.status_text = f"棋盘判定待确认 ({reason})，继续监控..."
                        self.current_state = "WAIT_BASE" if state == "VERIFY_POSITION" else "MONITOR"
                        self._sleep_cancelable(0.15)
                        continue

                    from_sq, to_sq = detected_move
                    candidate_uci = f"{from_sq}{to_sq}"
                    confirmed = self.move_confirmation.observe(detected_move, self._camera_sample_key)
                    self.last_detection_info['confirmation_frames'] = self.move_confirmation.count
                    self.last_detection_info['required_frames'] = self.move_confirmation.required_frames
                    self.last_detection_info['confirmed'] = confirmed
                    piece_name = PIECE_NAMES_CN.get(self.last_detection_info.get('piece'), '棋子')
                    if not confirmed:
                        self.status_text = (
                            f"候选{piece_name} {candidate_uci}，连续帧核验 "
                            f"{self.move_confirmation.count}/{self.move_confirmation.required_frames}..."
                        )
                        self._sleep_cancelable(0.2)
                        continue
                    if self.is_rotated:
                        from xiangqi_endgame import rotate_square_180
                        std_uci = f"{rotate_square_180(from_sq)}{rotate_square_180(to_sq)}"
                    else:
                        std_uci = candidate_uci

                    with self.lock:
                        if self.game_id != loop_game_id or not self.is_running:
                            break
                        if not self.board.is_legal_move(std_uci):
                            self.status_text = f"违规或未对齐着法 [{candidate_uci}]，继续等待正确走子..."
                            self.current_state = "MONITOR"
                            self.stable_duration = 0.0
                            continue
                        moving_piece = self.board.get_piece(std_uci[:2])
                        ok, captured_symbol = self.board.apply_move(std_uci)
                        if not ok:
                            self.current_state = "MONITOR"
                            continue
                        self.board_version += 1
                        self.base_frame = curr_frame.copy()
                        self._baseline_ignored_squares = ()
                        self.last_red_move = candidate_uci
                        self.move_history.append({
                            "side": "red",
                            "piece": moving_piece,
                            "piece_name": PIECE_NAMES_CN[moving_piece],
                            "move": candidate_uci,
                            "std_move": std_uci,
                            "captured": captured_symbol,
                            "fen": self.board.get_fen(),
                            "time": time.strftime("%H:%M:%S")
                        })
                        self.status_text = f"确认{PIECE_NAMES_CN[moving_piece]} {candidate_uci}，正在呼叫 Pikafish 引擎思考对策..."
                        self.current_state = "PIKAFISH"
                    continue

                elif state == "PIKAFISH":
                    with self.lock:
                        if self.game_id != loop_game_id or not self.is_running:
                            break
                        current_fen = self.board.get_fen()
                        b_ver = self.board_version
                    self.status_text = f"Pikafish 深度推演中 (FEN: {current_fen[:30]}...)..."
                    best_move, ponder = self.pikafish.get_best_move(current_fen, movetime_ms=500)

                    if not best_move or len(best_move) != 4:
                        self._enter_fault("PIKAFISH", RuntimeError(f"Pikafish 未输出有效着法: {best_move!r}"))
                        break

                    if self.is_rotated:
                        from xiangqi_endgame import rotate_square_180
                        phys_move = f"{rotate_square_180(best_move[:2])}{rotate_square_180(best_move[2:])}"
                    else:
                        phys_move = best_move

                    with self.lock:
                        if self.game_id != loop_game_id or self.board_version != b_ver or not self.is_running:
                            break
                        self.last_black_move = phys_move
                        self._last_black_std_move = best_move
                        self.status_text = f"Pikafish 决策完成: 黑方走【{phys_move}】，准备驱动机械臂执行..."
                        self.current_state = "ARM_MOVE"
                    continue

                elif state == "ARM_MOVE":
                    with self.lock:
                        b_ver = self.board_version
                    try:
                        self._execute_arm_move_phases(loop_game_id, b_ver)
                    except Exception as e:
                        self._enter_fault("ARM_MOVE", e, extra={"move": self.last_black_move})
                        break

                    self.status_text = f"黑方落子完成 ({self.last_black_move})，机械臂复位待机，等待红方..."
                    self.current_state = "WAIT_BASE"
                    self._baseline_home_ready = False
                    self.move_confirmation.reset()
                    self._sleep_cancelable(0.8)
                    continue

                else:
                    break

            except Exception as e:
                self._enter_fault(self.current_state, e)
                break

    def _get_fresh_camera_frame(self) -> Optional[np.ndarray]:
        """优先通过进程内 frame_provider 零拷贝取帧，未注入时回退 HTTP 拉帧。"""
        self._camera_sample_key = ('sample', time.monotonic_ns())
        if callable(self.frame_provider):
            try:
                res = self.frame_provider()
                if isinstance(res, tuple) and len(res) == 2:
                    frame, meta = res
                    if meta and not meta.get("fresh", True):
                        return None
                    if meta and meta.get('frame_id') is not None:
                        self._camera_sample_key = ('camera', meta['frame_id'])
                    elif meta and meta.get('timestamp') is not None:
                        self._camera_sample_key = ('capture', meta['timestamp'])
                    return frame
                return res
            except Exception:
                return None
        try:
            import requests
            r = requests.get("http://127.0.0.1:5000/api/raw_frame", timeout=1.5)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass
        return None

    def _detect_moved_square_pair(self, base_img: np.ndarray, curr_img: np.ndarray) -> Optional[Tuple[str, str]]:
        """先核实棋子离位与落位，再根据当前棋局校验着法。"""
        with self.lock:
            board = XiangqiBoard(self.board.get_fen())
            rotated = self.is_rotated
            version = self.board_version
        if base_img is None:
            move, info = self.move_detector.detect_from_position(curr_img, board, self.grid_mapper, rotated)
        else:
            move, info = self.move_detector.detect(
                base_img, curr_img, board, self.grid_mapper, rotated, self._baseline_ignored_squares,
            )
        with self.lock:
            if version != self.board_version:
                return None
            self.last_detection_info = info
        return move

    def _get_calib_offset(self) -> Tuple[float, float, float]:
        """实时读取手眼微调偏置 (ΔX, ΔY, ΔZ mm)"""
        raw = None
        provider = getattr(self, "offset_provider", None)
        if callable(provider):
            raw = provider()
        elif getattr(self, "grid_mapper", None) and hasattr(self.grid_mapper.hand_eye_mapper, "get_offset"):
            raw = self.grid_mapper.hand_eye_mapper.get_offset()
        elif getattr(self, "executor", None) and hasattr(self.executor.mapper, "get_offset"):
            raw = self.executor.mapper.get_offset()

        if raw and len(raw) >= 3:
            return (round(float(raw[0]), 1), round(float(raw[1]), 1), round(float(raw[2]), 1))
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _square_bbox(nx: float, ny: float, half_size: float = 40.0) -> List[float]:
        """构造以棋盘交叉点为中心、避免 int() 截断漂移的归一化包围盒"""
        return [
            round(max(0.0, ny - half_size), 2),
            round(max(0.0, nx - half_size), 2),
            round(min(1000.0, ny + half_size), 2),
            round(min(1000.0, nx + half_size), 2),
        ]

    def _run_square_pick_and_place(
        self,
        target_name: str,
        sq: str,
        target_station: Any,
        offset_trim: Tuple[float, float, float],
        cancel_token: Any = None,
    ):
        """执行单格棋子抓取与放置，并在被安全拦截或失败时显式抛出异常"""
        nx, ny = self.grid_mapper.get_square_norm_pos(sq)
        bbox = self._square_bbox(nx, ny)
        kwargs: Dict[str, Any] = {
            "target_name": target_name,
            "bbox": bbox,
            "target_station": target_station,
            "speed": 25,
            "offset_trim": offset_trim,
            "gripper_open_percent": CHESS_GRIPPER_OPEN_PERCENT,
            "release_at_hover": True,
        }
        if cancel_token is not None:
            kwargs["cancel_token"] = cancel_token
        res = self.executor.execute_visual_pick_and_place(**kwargs)
        if isinstance(res, dict) and res.get("success") is False:
            raise RuntimeError(res.get("error", f"执行 {target_name} 抓放失败"))
        return res

    def _physical_pick_and_place_square(self, sq: str, dest_station: str = "P5", cancel_token: Any = None):
        """将指定格子的棋子拾取并移至预设工位 (如弃子盒 P5)"""
        offset_trim = self._get_calib_offset()
        print(
            f"♟️ [象棋拾取] 格子={sq} -> 工位={dest_station}, "
            f"手眼微调偏置=({offset_trim[0]:+.1f}, {offset_trim[1]:+.1f}, {offset_trim[2]:+.1f})mm"
        )
        self._run_square_pick_and_place(f"棋子_{sq}", sq, dest_station, offset_trim, cancel_token=cancel_token)

    def _physical_move_piece(self, from_sq: str, to_sq: str, cancel_token: Any = None):
        """将棋子从 from_sq 移动至 to_sq（拾取与落子点均应用当前手眼微调偏置）"""
        offset_trim = self._get_calib_offset()
        to_xyz = self.grid_mapper.get_square_robot_xyz(to_sq, offset_trim=offset_trim)
        print(
            f"♟️ [象棋落子] {from_sq} -> {to_sq} 目标={to_xyz}, "
            f"手眼微调偏置=({offset_trim[0]:+.1f}, {offset_trim[1]:+.1f}, {offset_trim[2]:+.1f})mm"
        )
        # to_xyz 已包含 offset_trim；execute_visual_pick_and_place 仅将 offset_trim 作用于拾取点，不会二次叠加
        self._run_square_pick_and_place(f"黑子_{from_sq}", from_sq, to_xyz, offset_trim, cancel_token=cancel_token)
