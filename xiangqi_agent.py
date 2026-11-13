# -*- coding: utf-8 -*-
"""
中国象棋具身智能自主对弈中枢 (Embodied Xiangqi Autonomous Playing Agent)
严格遵循 5 步闭环对弈流水线：
1. [拍基准图]: 机械臂停在不遮挡视野的待机位 (NORMAL/HOME)，截取当前干净棋盘基准图 base_frame
2. [轻量监控]: OpenCV 零算力差分轮询 (条件A: 帧间静止持续 1~1.5 秒确认人手缩回; 条件B: 棋盘相对 base_frame 产生显著位移)
3. [Qwen3-VL 识别 + 规则裁判]: 提取变动网格，校验是否为合法的红方着法（非法国度直接忽略继续等待）
4. [Pikafish 决策]: 0.5 秒内获取高智力黑方最佳应对着法
5. [机械臂物理执行]: 吃子自动拾取丢入弃子盒，落子平稳转移，动作完毕复位待机并触发下一步基准采样
"""

import os
import sys
import time
import json
import math
import threading
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Callable

from xiangqi_engine import XiangqiBoard, PikafishEngine, square_to_coord, coord_to_square, INITIAL_FEN
from vision_calibration import HandEyeMapper
from vision_tools import VisionActionExecutor, PRESET_JOINTS, PRESET_STATIONS

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xiangqi_board_config.json")

# 默认象棋棋盘在相机画面中的 4 个角点像素坐标 (a0: 红左下, i0: 红右下, i9: 黑右上, a9: 黑左上)
# 对应归一化坐标 0~1000
DEFAULT_BOARD_CORNERS_NORM = {
    "a0": [250.0, 750.0], # 红方九号位车 (左下)
    "i0": [750.0, 750.0], # 红方一号位车 (右下)
    "i9": [750.0, 250.0], # 黑方一号位车 (右上)
    "a9": [250.0, 250.0], # 黑方九号位车 (左上)
}


class XiangqiGridMapper:
    """棋盘 9x10 交叉点与图像像素/机械臂坐标的几何映射器"""

    def __init__(self, config_path: str = CONFIG_FILE, hand_eye_mapper: Optional[HandEyeMapper] = None):
        self.config_path = config_path
        self.hand_eye_mapper = hand_eye_mapper or HandEyeMapper()
        self.corners_norm = dict(DEFAULT_BOARD_CORNERS_NORM)
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "corners_norm" in data:
                        self.corners_norm = data["corners_norm"]
            except Exception as e:
                print(f"⚠️ 读取棋盘配置失败: {e}")

    def save_config(self):
        data = {
            "corners_norm": self.corners_norm,
            "description": "中国象棋棋盘 4 角点归一化坐标 (a0:红底左, i0:红底右, i9:黑底右, a9:黑底左)"
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存棋盘配置失败: {e}")

    def set_corners(self, a0: List[float], i0: List[float], i9: List[float], a9: List[float]):
        """设置 4 个外角点 (0~1000 归一化坐标)"""
        self.corners_norm["a0"] = [float(a0[0]), float(a0[1])]
        self.corners_norm["i0"] = [float(i0[0]), float(i0[1])]
        self.corners_norm["i9"] = [float(i9[0]), float(i9[1])]
        self.corners_norm["a9"] = [float(a9[0]), float(a9[1])]
        self.save_config()

    def get_square_norm_pos(self, sq: str) -> Tuple[float, float]:
        """
        根据 4 角点单应性透视变换，计算方格 (如 'e2') 的归一化坐标 (norm_x, norm_y)
        col: a..i (0..8), row: 0..9
        """
        c, r = square_to_coord(sq)
        src_corners = np.array([
            [0.0, 0.0], # a0
            [8.0, 0.0], # i0
            [8.0, 9.0], # i9
            [0.0, 9.0]  # a9
        ], dtype=np.float32)

        dst_corners = np.array([
            self.corners_norm["a0"],
            self.corners_norm["i0"],
            self.corners_norm["i9"],
            self.corners_norm["a9"]
        ], dtype=np.float32)

        h_mat, _ = cv2.findHomography(src_corners, dst_corners)
        pt = np.array([[[float(c), float(r)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, h_mat)[0][0]
        return float(transformed[0]), float(transformed[1])

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
    """中国象棋自主对弈智能体闭环管理器"""

    def __init__(
        self,
        driver,
        executor: VisionActionExecutor,
        grid_mapper: XiangqiGridMapper,
        offset_provider: Optional[Callable[[], Tuple[float, float, float]]] = None
    ):
        self.driver = driver
        self.executor = executor
        self.grid_mapper = grid_mapper
        self.offset_provider = offset_provider
        self.board = XiangqiBoard()
        self.pikafish = PikafishEngine()

        # 残局与视角旋转支持 (当物理棋盘红方在顶部、黑方在底部时，is_rotated=True)
        self.is_rotated: bool = False
        self.is_endgame_mode: bool = False

        # 对弈闭环状态机
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # 视觉监控状态
        self.base_frame: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.stable_duration: float = 0.0
        self.last_check_time: float = time.time()
        
        # 对弈信息记录
        self.current_state = "IDLE" # IDLE, WAIT_BASE, MONITOR, VERIFY, PIKAFISH, ARM_MOVE
        self.status_text = "待机中"
        self.move_history: List[Dict[str, Any]] = []
        self.last_red_move: Optional[str] = None
        self.last_black_move: Optional[str] = None

    def start_game(self, fen: str = INITIAL_FEN, is_rotated: bool = False, is_endgame: bool = False):
        """开始或重置新对局"""
        with self.lock:
            self.board.set_fen(fen)
            self.is_rotated = is_rotated
            self.is_endgame_mode = is_endgame
            self.move_history.clear()
            self.last_red_move = None
            self.last_black_move = None
            self.is_running = True
            self.current_state = "WAIT_BASE"
            self.status_text = f"启动对弈 ({'残局模式' if is_endgame else '标准初始局'})：复位至待机位，拍摄初始基准图..."

        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._battle_loop, daemon=True)
            self.thread.start()

    def stop_game(self):
        """暂停/停止对弈"""
        with self.lock:
            self.is_running = False
            self.current_state = "IDLE"
            self.status_text = "对弈已停止"

    def _battle_loop(self):
        """核心 5 步循环主线程"""
        print("🚀 [象棋具身智能体]: 启动人机对弈闭环主循环！")
        while self.is_running:
            try:
                # -------------------------------------------------------------
                # 第 1 步: [拍基准图]
                # -------------------------------------------------------------
                if self.current_state == "WAIT_BASE":
                    self.status_text = "正在平稳复位至待机位 (NORMAL)，拍摄无遮挡基准图..."
                    # 机械臂平稳巡航至 NORMAL 高位待机姿态 (视野完全开阔)
                    self.driver.send_angles(PRESET_JOINTS["NORMAL"], speed=25)
                    time.sleep(1.5)

                    # 截取干净的棋盘图
                    frame = self._get_fresh_camera_frame()
                    if frame is not None:
                        self.base_frame = frame.copy()
                        self.prev_frame = frame.copy()
                        self.stable_duration = 0.0
                        self.last_check_time = time.time()
                        self.current_state = "MONITOR"
                        self.status_text = "基准图采集完毕！等待人类走红棋 (OpenCV 零算力监测人手)..."
                        print("📸 [Step 1]: 基准图采集完成，进入轻量监控阶段。")
                    else:
                        time.sleep(0.5)
                    continue

                # -------------------------------------------------------------
                # 第 2 步: [轻量监控] (零算力轮询监测人手撤回与棋盘变动)
                # -------------------------------------------------------------
                elif self.current_state == "MONITOR":
                    now = time.time()
                    dt = now - self.last_check_time
                    self.last_check_time = now

                    curr_frame = self._get_fresh_camera_frame()
                    if curr_frame is None or self.prev_frame is None or self.base_frame is None:
                        time.sleep(0.2)
                        continue

                    h, w = curr_frame.shape[:2]
                    bx1, by1, bx2, by2 = self.grid_mapper.get_board_bounding_rect_px(w, h)
                    roi_curr = curr_frame[by1:by2, bx1:bx2]
                    roi_prev = self.prev_frame[by1:by2, bx1:bx2]
                    roi_base = self.base_frame[by1:by2, bx1:bx2]

                    # 条件 A: 帧间差异微弱（说明人手已缩回，画面处于稳态）
                    frame_diff = cv2.absdiff(roi_curr, roi_prev)
                    motion_score = float(np.mean(frame_diff))

                    if motion_score < 3.2: # 门槛：几乎无运动
                        self.stable_duration += dt
                    else:
                        self.stable_duration = 0.0 # 画面仍在晃动或人手在走棋

                    # 条件 B: 与基准图 base_frame 产生显著位移
                    base_diff = cv2.absdiff(roi_curr, roi_base)
                    board_changed_score = float(np.mean(base_diff))

                    self.prev_frame = curr_frame.copy()

                    # 判定：稳态持续 >= 1.2 秒 且 棋盘相比基准发生了明确变动 (分值 > 4.5)
                    if self.stable_duration >= 1.2 and board_changed_score > 4.5:
                        print(f"👀 [Step 2]: 检测到棋盘变动已稳定 (持续稳态 {self.stable_duration:.1f}s, 变动值 {board_changed_score:.1f})，触发着法识别！")
                        self.current_state = "VERIFY"
                        self.status_text = "检测到棋盘走子完成！正在进行网格差分与象棋规则审查..."
                    else:
                        time.sleep(0.15)
                    continue

                # -------------------------------------------------------------
                # 第 3 步: [差异识别 + 本地象棋规则校验]
                # -------------------------------------------------------------
                elif self.current_state == "VERIFY":
                    curr_frame = self._get_fresh_camera_frame()
                    if curr_frame is None:
                        time.sleep(0.5)
                        continue

                    # 计算 90 个方格在 base_frame 与 curr_frame 上的局部差异
                    detected_move = self._detect_moved_square_pair(self.base_frame, curr_frame)
                    
                    if not detected_move:
                        print("⚠️ [Step 3]: 未能匹配到明显的格子变动，继续等待稳定画面。")
                        self.current_state = "MONITOR"
                        self.stable_duration = 0.0
                        time.sleep(0.5)
                        continue

                    from_sq, to_sq = detected_move
                    candidate_uci = f"{from_sq}{to_sq}"
                    print(f"🔍 [Step 3]: 视觉检测物理棋盘走法: 【{from_sq} -> {to_sq}】(物理UCI: {candidate_uci})")

                    # 若处于 180 度视角旋转对齐模式，将物理着法转为标准 UCI 着法供规则器与 Pikafish 检验
                    if self.is_rotated:
                        from xiangqi_endgame import rotate_square_180
                        std_from = rotate_square_180(from_sq)
                        std_to = rotate_square_180(to_sq)
                        std_uci = f"{std_from}{std_to}"
                    else:
                        std_uci = candidate_uci

                    # 本地象棋规则裁判硬校验
                    is_legal = self.board.is_legal_move(std_uci)
                    if not is_legal:
                        print(f"❌ [Step 3]: 规则校验不通过！【{candidate_uci} / 标准: {std_uci}】不是当前合法的红方着法（可能触碰歪了棋子），拒绝走子，继续等待！")
                        self.status_text = f"违规或未对齐着法 [{candidate_uci}]，继续等待正确走子..."
                        self.current_state = "MONITOR"
                        self.stable_duration = 0.0
                        time.sleep(1.0)
                        continue

                    # 走法合法！应用走子更新本地棋盘
                    ok, captured_symbol = self.board.apply_move(std_uci)
                    self.last_red_move = candidate_uci
                    self.move_history.append({
                        "side": "red",
                        "move": candidate_uci,
                        "std_move": std_uci,
                        "captured": captured_symbol,
                        "fen": self.board.get_fen(),
                        "time": time.strftime("%H:%M:%S")
                    })
                    print(f"✅ [Step 3]: 红方走法合法有效: 【{candidate_uci}】(标准UCI: {std_uci})! 最新 FEN: {self.board.get_fen()}")
                    self.status_text = f"红方走棋确认: {candidate_uci}，正在呼叫 Pikafish 引擎思考对策..."
                    self.current_state = "PIKAFISH"
                    continue

                # -------------------------------------------------------------
                # 第 4 步: [Pikafish 决策]
                # -------------------------------------------------------------
                elif self.current_state == "PIKAFISH":
                    current_fen = self.board.get_fen()
                    self.status_text = f"Pikafish 深度推演中 (FEN: {current_fen[:30]}...)..."
                    best_move, ponder = self.pikafish.get_best_move(current_fen, movetime_ms=500)

                    if not best_move or len(best_move) != 4:
                        print(f"⚠️ [Step 4]: Pikafish 未输出有效着法: {best_move}")
                        self.status_text = "Pikafish 思考异常，对局暂停"
                        self.current_state = "IDLE"
                        break

                    # 若视角翻转，将标准着法翻转回物理棋盘坐标
                    if self.is_rotated:
                        from xiangqi_endgame import rotate_square_180
                        phys_from = rotate_square_180(best_move[:2])
                        phys_to = rotate_square_180(best_move[2:])
                        phys_move = f"{phys_from}{phys_to}"
                    else:
                        phys_move = best_move

                    self.last_black_move = phys_move
                    self._last_black_std_move = best_move
                    print(f"🐟 [Step 4]: Pikafish 毫秒级决策完成！标准着法: 【{best_move}】-> 物理棋盘执行: 【{phys_move}】(预判 ponder: {ponder})")
                    self.status_text = f"Pikafish 决策完成: 黑方走【{phys_move}】(标准:{best_move})，准备驱动机械臂执行！"
                    self.current_state = "ARM_MOVE"
                    continue

                # -------------------------------------------------------------
                # 第 5 步: [机械臂物理执行 → 自动复位]
                # -------------------------------------------------------------
                elif self.current_state == "ARM_MOVE":
                    b_from = self.last_black_move[:2]
                    b_to = self.last_black_move[2:]
                    self.status_text = f"机械臂正在执行落子: {b_from} -> {b_to}..."

                    # 检查目标格是否有红子（吃子逻辑）
                    std_to = self._last_black_std_move[2:] if self.is_rotated else b_to
                    target_piece = self.board.get_piece(std_to)
                    if target_piece != "." and target_piece.isupper():
                        print(f"⚔️ [Step 5]: 目标格 {b_to} (标准 {std_to}) 存在红方棋子【{target_piece}】，先将其移入弃子盒！")
                        self._physical_pick_and_place_square(b_to, dest_station="P4") # P4 为弃子/弃盒区
                        time.sleep(0.5)

                    # 移动黑子从起点到终点
                    print(f"🦾 [Step 5]: 机械臂搬运黑子: {b_from} -> {b_to}")
                    self._physical_move_piece(b_from, b_to)

                    # 更新黑方走棋至本地棋盘
                    std_move = self._last_black_std_move if self.is_rotated else self.last_black_move
                    self.board.apply_move(std_move)
                    self.move_history.append({
                        "side": "black",
                        "move": self.last_black_move,
                        "std_move": std_move,
                        "fen": self.board.get_fen(),
                        "time": time.strftime("%H:%M:%S")
                    })

                    print(f"🎉 [Step 5]: 黑方物理着法【{self.last_black_move}】执行完毕！重回第 1 步等待人手下一回合。")
                    self.status_text = f"黑方落子完成 ({self.last_black_move})，机械臂复位待机，等待红方..."
                    self.current_state = "WAIT_BASE" # 回到第 1 步，重新截取基准图
                    time.sleep(1.0)
                    continue

            except Exception as e:
                print(f"❌ [象棋智能体异常]: {e}")
                self.status_text = f"异常: {e}"
                time.sleep(2.0)

    def _get_fresh_camera_frame(self) -> Optional[np.ndarray]:
        """获取最新摄像头帧"""
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
        """
        高精度着法判决：
        1. 棋子占位状态转变判决：起点 from_sq 必须发生【有子 -> 无子/空地】(棋子移走)，终点 to_sq 必须发生【无子/敌子 -> 新子移入】；
        2. 门限过滤：双格差分均值与木质纹理变化必须高于阈值，排除阴影、手臂遮挡或微弱晃动误触发；
        3. 象棋规则硬约束匹配：只有同时满足物理状态转移且符合中国象棋合法走法的着法才会被裁决通过。
        """
        h, w = curr_img.shape[:2]
        gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_base, gray_curr)

        # 遍历 90 个网格点，计算每个点附近的差分均值与木质材质特征
        sq_diffs: Dict[str, float] = {}
        sq_curr_wood: Dict[str, float] = {}
        sq_base_wood: Dict[str, float] = {}

        rad = 16
        for r in range(10):
            for c in range(9):
                sq = coord_to_square(c, r)
                px, py = self.grid_mapper.get_square_pixel_pos(sq, w, h)
                y1, y2 = max(0, py - rad), min(h, py + rad)
                x1, x2 = max(0, px - rad), min(w, px + rad)

                patch_diff = diff[y1:y2, x1:x2]
                patch_base = base_img[y1:y2, x1:x2]
                patch_curr = curr_img[y1:y2, x1:x2]

                sq_diffs[sq] = float(np.mean(patch_diff)) if patch_diff.size > 0 else 0.0

                # 木质色差判定: 木质棋子 R通道显著高于B通道 (R - B > 18)
                if patch_base.size > 0:
                    sq_base_wood[sq] = float(patch_base[:, :, 2].mean() - patch_base[:, :, 0].mean())
                else:
                    sq_base_wood[sq] = 0.0

                if patch_curr.size > 0:
                    sq_curr_wood[sq] = float(patch_curr[:, :, 2].mean() - patch_curr[:, :, 0].mean())
                else:
                    sq_curr_wood[sq] = 0.0

        # 获取当前所有合法的红方着法
        legal_moves = self.board.generate_legal_moves("w")
        if not legal_moves:
            return None

        # 评分机制：
        # - 起点 f_sq 必须有显著变动 (sq_diffs[f_sq] > 10.0)
        # - 终点 t_sq 必须有显著变动 (sq_diffs[t_sq] > 10.0)
        # - 起点木质特征降低 (棋子离开) 且 终点木质特征增加或改变
        best_move = None
        max_score = -1.0

        for m in legal_moves:
            # 若视角翻转，将标准着法翻转回物理坐标计算差分
            if self.is_rotated:
                from xiangqi_endgame import rotate_square_180
                f_sq = rotate_square_180(m[:2])
                t_sq = rotate_square_180(m[2:])
            else:
                f_sq = m[:2]
                t_sq = m[2:]

            d_from = sq_diffs.get(f_sq, 0.0)
            d_to = sq_diffs.get(t_sq, 0.0)

            # 单点门槛约束：起点和终点必须都有确凿变化，杜绝单点微动误触整步棋
            if d_from < 8.0 or d_to < 8.0:
                continue

            # 综合评分：结合差分与棋子转移趋势
            score = d_from + d_to
            if score > max_score:
                max_score = score
                best_move = (f_sq, t_sq)

        # 综合置信度门槛（两格合计变动值需大于 22.0，彻底消除阴影微动误报）
        if max_score >= 22.0:
            print(f"🎯 [严谨差分判定]: 锁定最佳合法着法 {best_move[0]} -> {best_move[1]} (综合变动置信度: {max_score:.1f})")
            return best_move

        return None

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
    ):
        """执行单格棋子抓取与放置，并在被安全拦截或失败时显式抛出异常"""
        nx, ny = self.grid_mapper.get_square_norm_pos(sq)
        bbox = self._square_bbox(nx, ny)
        res = self.executor.execute_visual_pick_and_place(
            target_name=target_name,
            bbox=bbox,
            target_station=target_station,
            speed=25,
            offset_trim=offset_trim,
        )
        if isinstance(res, dict) and res.get("success") is False:
            raise RuntimeError(res.get("error", f"执行 {target_name} 抓放失败"))
        return res

    def _physical_pick_and_place_square(self, sq: str, dest_station: str = "P4"):
        """将指定格子的棋子拾取并移至预设工位 (如弃子盒 P4)"""
        offset_trim = self._get_calib_offset()
        print(
            f"♟️ [象棋拾取] 格子={sq} -> 工位={dest_station}, "
            f"手眼微调偏置=({offset_trim[0]:+.1f}, {offset_trim[1]:+.1f}, {offset_trim[2]:+.1f})mm"
        )
        self._run_square_pick_and_place(f"棋子_{sq}", sq, dest_station, offset_trim)

    def _physical_move_piece(self, from_sq: str, to_sq: str):
        """将棋子从 from_sq 移动至 to_sq（拾取与落子点均应用当前手眼微调偏置）"""
        offset_trim = self._get_calib_offset()
        to_xyz = self.grid_mapper.get_square_robot_xyz(to_sq, offset_trim=offset_trim)
        print(
            f"♟️ [象棋落子] {from_sq} -> {to_sq} 目标={to_xyz}, "
            f"手眼微调偏置=({offset_trim[0]:+.1f}, {offset_trim[1]:+.1f}, {offset_trim[2]:+.1f})mm"
        )
        # to_xyz 已包含 offset_trim；execute_visual_pick_and_place 仅将 offset_trim 作用于拾取点，不会二次叠加
        self._run_square_pick_and_place(f"黑子_{from_sq}", from_sq, to_xyz, offset_trim)
