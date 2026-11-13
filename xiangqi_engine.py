# -*- coding: utf-8 -*-
"""
中国象棋完整规则引擎与 Pikafish 联动中枢 (Xiangqi Rule Engine & Pikafish Controller)
包含：
1. 完整中国象棋着法生成与合法性检验 (车马炮兵卒相象士仕将帅全部走子规则、别马腿、塞象眼、九宫格限制、过河规则、飞将对脸检测)
2. FEN 字符串生成与解析
3. 棋盘网格与物理坐标换算 (a0~i9 对应 90 个交叉点)
4. Pikafish UCI 引擎进程异步驱动与毫秒级决策
"""

import os
import sys
import time
import queue
import threading
import subprocess
from typing import List, Tuple, Optional, Dict, Any

# 棋子定义 (红方大写，黑方小写)
# Red:   R=车, N=马, B=相, A=仕, K=帅, C=炮, P=兵
# Black: r=车, n=马, b=象, a=士, k=将, c=炮, p=卒
INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
VALID_PIECE_CHARS = set("RNBAKCP rnbakcp".replace(" ", ""))

PIECE_NAMES_CN = {
    "R": "红车", "N": "红马", "B": "红相", "A": "红仕", "K": "红帅", "C": "红炮", "P": "红兵",
    "r": "黑车", "n": "黑马", "b": "黑象", "a": "黑士", "k": "黑将", "c": "黑炮", "p": "黑卒",
    ".": "空"
}

FILES = "abcdefghi" # 0 to 8
RANKS = "0123456789" # 0 to 9

def is_valid_square(sq: str) -> bool:
    return isinstance(sq, str) and len(sq) == 2 and sq[0].lower() in FILES and sq[1] in RANKS

def square_to_coord(sq: str) -> Tuple[int, int]:
    """将 UCI 方格 (如 'b2') 转换为 (col, row), col 0..8, row 0..9"""
    if not is_valid_square(sq):
        raise ValueError(f"非法棋盘坐标: {sq!r}")
    c = ord(sq[0].lower()) - ord('a')
    r = int(sq[1])
    return c, r

def coord_to_square(col: int, row: int) -> str:
    """将 (col, row) 转换为 UCI 方格 (如 (1, 2) -> 'b2')"""
    return f"{FILES[col]}{row}"


class XiangqiBoard:
    """中国象棋棋盘与走法合法性裁判"""

    def __init__(self, fen: str = INITIAL_FEN):
        self.board: List[List[str]] = [["." for _ in range(9)] for _ in range(10)]
        self.turn: str = "w" # 'w' for Red, 'b' for Black
        self.set_fen(fen)

    @staticmethod
    def validate_fen(fen: str) -> Tuple[bool, Optional[str]]:
        """严格校验中国象棋 FEN 格式与将帅唯一性/九宫约束。"""
        if not isinstance(fen, str) or not fen.strip():
            return False, "FEN 不能为空"
        parts = fen.strip().split()
        rows = parts[0].split("/")
        if len(rows) != 10:
            return False, f"FEN 行数必须为 10，当前为 {len(rows)}"
        red_kings: List[Tuple[int, int]] = []
        black_kings: List[Tuple[int, int]] = []
        for r_idx, row_str in enumerate(rows):
            board_r = 9 - r_idx
            c_idx = 0
            for ch in row_str:
                if "1" <= ch <= "9":
                    c_idx += int(ch)
                elif ch in VALID_PIECE_CHARS:
                    if c_idx >= 9:
                        return False, f"FEN 第 {r_idx + 1} 行列数超过 9"
                    if ch == "K":
                        red_kings.append((c_idx, board_r))
                    elif ch == "k":
                        black_kings.append((c_idx, board_r))
                    c_idx += 1
                else:
                    return False, f"FEN 包含非法字符: {ch!r}"
            if c_idx != 9:
                return False, f"FEN 第 {r_idx + 1} 行总列数为 {c_idx}，必须等于 9"
        if len(red_kings) != 1 or len(black_kings) != 1:
            return False, f"必须恰好有 1 个红帅和 1 个黑将 (当前红={len(red_kings)}, 黑={len(black_kings)})"
        rkc, rkr = red_kings[0]
        bkc, bkr = black_kings[0]
        if not (3 <= rkc <= 5 and 0 <= rkr <= 2):
            return False, f"红帅位置不在红方九宫内: {coord_to_square(rkc, rkr)}"
        if not (3 <= bkc <= 5 and 7 <= bkr <= 9):
            return False, f"黑将位置不在黑方九宫内: {coord_to_square(bkc, bkr)}"
        turn = parts[1] if len(parts) > 1 else "w"
        if turn not in ("w", "b"):
            return False, f"非法行棋方标记: {turn!r} (应为 'w' 或 'b')"
        return True, None

    def set_fen(self, fen: str):
        """解析并校验 FEN 串加载棋盘"""
        ok, err = self.validate_fen(fen)
        if not ok:
            raise ValueError(f"无效 FEN: {err}")
        parts = fen.strip().split()
        rows = parts[0].split("/")
        new_board = [["." for _ in range(9)] for _ in range(10)]
        # FEN row 0 是黑方底线 (rank 9)，row 9 是红方底线 (rank 0)
        for r_idx, row_str in enumerate(rows):
            board_r = 9 - r_idx
            c_idx = 0
            for ch in row_str:
                if ch.isdigit():
                    c_idx += int(ch)
                else:
                    new_board[board_r][c_idx] = ch
                    c_idx += 1
        self.board = new_board
        self.turn = parts[1] if len(parts) > 1 else "w"

    def get_fen(self) -> str:
        """生成当前棋盘的标准 FEN 串"""
        fen_rows = []
        for board_r in range(9, -1, -1):
            row_str = ""
            empty_count = 0
            for c_idx in range(9):
                p = self.board[board_r][c_idx]
                if p == ".":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += p
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)
        fen_board = "/".join(fen_rows)
        return f"{fen_board} {self.turn} - - 0 1"

    def get_piece(self, sq: str) -> str:
        col, row = square_to_coord(sq)
        return self.board[row][col]

    def is_red_piece(self, p: str) -> bool:
        return p.isupper()

    def is_black_piece(self, p: str) -> bool:
        return p.islower()

    def generate_legal_moves(self, side: Optional[str] = None) -> List[str]:
        """生成指定方 (默认当前方 'w' 或 'b') 的所有合法着法 (UCI 格式如 'b2e2')"""
        side = side or self.turn
        is_red = (side == "w")
        moves = []

        for r in range(10):
            for c in range(9):
                p = self.board[r][c]
                if p == ".":
                    continue
                if is_red and not self.is_red_piece(p):
                    continue
                if not is_red and not self.is_black_piece(p):
                    continue
                from_sq = coord_to_square(c, r)
                pseudo = self._pseudo_moves_for_piece(c, r, p)
                for to_sq in pseudo:
                    if self._is_move_safe_from_check(from_sq, to_sq, is_red):
                        moves.append(f"{from_sq}{to_sq}")
        return moves

    def _pseudo_moves_for_piece(self, c: int, r: int, p: str) -> List[str]:
        """伪合法着法生成 (未考虑送将和对脸帅)"""
        res = []
        is_red = p.isupper()
        ptype = p.upper()

        def is_friend(tc, tr):
            tp = self.board[tr][tc]
            if tp == ".":
                return False
            return self.is_red_piece(tp) if is_red else self.is_black_piece(tp)

        # 1. 车 (R/r)
        if ptype == "R":
            for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nc, nr = c + dc, r + dr
                while 0 <= nc <= 8 and 0 <= nr <= 9:
                    tp = self.board[nr][nc]
                    if tp == ".":
                        res.append(coord_to_square(nc, nr))
                    else:
                        if not is_friend(nc, nr):
                            res.append(coord_to_square(nc, nr))
                        break
                    nc += dc
                    nr += dr

        # 2. 炮 (C/c)
        elif ptype == "C":
            for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nc, nr = c + dc, r + dr
                over_screen = False
                while 0 <= nc <= 8 and 0 <= nr <= 9:
                    tp = self.board[nr][nc]
                    if not over_screen:
                        if tp == ".":
                            res.append(coord_to_square(nc, nr))
                        else:
                            over_screen = True
                    else:
                        if tp != ".":
                            if not is_friend(nc, nr):
                                res.append(coord_to_square(nc, nr))
                            break
                    nc += dc
                    nr += dr

        # 3. 马 (N/n) - 别马腿
        elif ptype == "N":
            jumps = [
                ((0, 1), [(1, 2), (-1, 2)]),   # 向上跳，腿在 (0, 1)
                ((0, -1), [(1, -2), (-1, -2)]), # 向下跳，腿在 (0, -1)
                ((1, 0), [(2, 1), (2, -1)]),   # 向右跳，腿在 (1, 0)
                ((-1, 0), [(-2, 1), (-2, -1)])  # 向左跳，腿在 (-1, 0)
            ]
            for (leg_dc, leg_dr), targets in jumps:
                leg_c, leg_r = c + leg_dc, r + leg_dr
                if 0 <= leg_c <= 8 and 0 <= leg_r <= 9:
                    if self.board[leg_r][leg_c] == ".": # 马腿无阻碍
                        for tc_delta, tr_delta in targets:
                            tc, tr = c + tc_delta, r + tr_delta
                            if 0 <= tc <= 8 and 0 <= tr <= 9 and not is_friend(tc, tr):
                                res.append(coord_to_square(tc, tr))

        # 4. 相/象 (B/b) - 塞象眼与不过河
        elif ptype == "B":
            eyes = [
                ((1, 1), (2, 2)),
                ((-1, 1), (-2, 2)),
                ((1, -1), (2, -2)),
                ((-1, -1), (-2, -2))
            ]
            for (ec, er), (tc_d, tr_d) in eyes:
                eye_c, eye_r = c + ec, r + er
                tc, tr = c + tc_d, r + tr_d
                if 0 <= eye_c <= 8 and 0 <= eye_r <= 9 and 0 <= tc <= 8 and 0 <= tr <= 9:
                    # 不能过河: 红相 rank 0..4，黑象 rank 5..9
                    if is_red and tr > 4:
                        continue
                    if not is_red and tr < 5:
                        continue
                    if self.board[eye_r][eye_c] == "." and not is_friend(tc, tr):
                        res.append(coord_to_square(tc, tr))

        # 5. 士/仕 (A/a) - 九宫格内斜行 1 格
        elif ptype == "A":
            for dc, dr in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                tc, tr = c + dc, r + dr
                if 3 <= tc <= 5:
                    if (is_red and 0 <= tr <= 2) or (not is_red and 7 <= tr <= 9):
                        if not is_friend(tc, tr):
                            res.append(coord_to_square(tc, tr))

        # 6. 将/帅 (K/k) - 九宫格内直行 1 格
        elif ptype == "K":
            for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                tc, tr = c + dc, r + dr
                if 3 <= tc <= 5:
                    if (is_red and 0 <= tr <= 2) or (not is_red and 7 <= tr <= 9):
                        if not is_friend(tc, tr):
                            res.append(coord_to_square(tc, tr))

        # 7. 兵/卒 (P/p)
        elif ptype == "P":
            if is_red:
                # 前进 (r+1)
                if r + 1 <= 9 and not is_friend(c, r + 1):
                    res.append(coord_to_square(c, r + 1))
                # 过河后可横移 (r >= 5)
                if r >= 5:
                    for dc in [-1, 1]:
                        if 0 <= c + dc <= 8 and not is_friend(c + dc, r):
                            res.append(coord_to_square(c + dc, r))
            else:
                # 前进 (r-1)
                if r - 1 >= 0 and not is_friend(c, r - 1):
                    res.append(coord_to_square(c, r - 1))
                # 过河后可横移 (r <= 4)
                if r <= 4:
                    for dc in [-1, 1]:
                        if 0 <= c + dc <= 8 and not is_friend(c + dc, r):
                            res.append(coord_to_square(c + dc, r))

        return res

    def is_square_attacked(self, target_c: int, target_r: int, by_red: bool) -> bool:
        """完整检测目标格是否正被指定一方 (by_red=True 为红方) 的任意棋子攻击（车/炮/马+别腿/兵卒/将帅）。"""
        enemy_r = "R" if by_red else "r"
        enemy_c = "C" if by_red else "c"
        enemy_n = "N" if by_red else "n"
        enemy_p = "P" if by_red else "p"
        enemy_k = "K" if by_red else "k"

        # 1. 正交射线：车（0 阻挡）、炮（恰 1 炮架）、对脸飞将（同列 0 阻挡）
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = target_c + dc, target_r + dr
            screen_seen = False
            while 0 <= nc <= 8 and 0 <= nr <= 9:
                p = self.board[nr][nc]
                if p != ".":
                    if not screen_seen:
                        if p == enemy_r or p == enemy_k:
                            if p == enemy_r:
                                return True
                            # 将帅只有同列无阻挡时形成对脸攻击
                            if dc == 0:
                                return True
                        screen_seen = True
                    else:
                        if p == enemy_c:
                            return True
                        break
                nc += dc
                nr += dr

        # 2. 马（检测别马腿：进攻马位置为 (ac, ar)，其阻挡腿位置为 (target_c + sgn(dc), target_r + sgn(dr))）
        for dc, dr in ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
            ac, ar = target_c + dc, target_r + dr
            if 0 <= ac <= 8 and 0 <= ar <= 9 and self.board[ar][ac] == enemy_n:
                leg_c = target_c + (1 if dc > 0 else -1)
                leg_r = target_r + (1 if dr > 0 else -1)
                if self.board[leg_r][leg_c] == ".":
                    return True

        # 3. 兵 / 卒
        if by_red:
            # 红兵由下往上 (+r)，在 r>=5 时可左右平走
            if target_r - 1 >= 0 and self.board[target_r - 1][target_c] == enemy_p:
                return True
            if target_r >= 5:
                for dc in (-1, 1):
                    if 0 <= target_c + dc <= 8 and self.board[target_r][target_c + dc] == enemy_p:
                        return True
        else:
            # 黑卒由上往下 (-r)，在 r<=4 时可左右平走
            if target_r + 1 <= 9 and self.board[target_r + 1][target_c] == enemy_p:
                return True
            if target_r <= 4:
                for dc in (-1, 1):
                    if 0 <= target_c + dc <= 8 and self.board[target_r][target_c + dc] == enemy_p:
                        return True

        return False

    def is_in_check(self, is_red: Optional[bool] = None) -> bool:
        """检查指定方（默认当前行棋方）老将/帅是否正在被将军或对脸。"""
        if is_red is None:
            is_red = (self.turn == "w")
        king_sym = "K" if is_red else "k"
        for r in range(10):
            for c in range(9):
                if self.board[r][c] == king_sym:
                    return self.is_square_attacked(c, r, by_red=not is_red)
        return True

    def _is_move_safe_from_check(self, from_sq: str, to_sq: str, is_red: bool) -> bool:
        """检查执行移动后是否仍安全（不送将、不留应将、不飞将对脸）。"""
        fc, fr = square_to_coord(from_sq)
        tc, tr = square_to_coord(to_sq)
        orig_p = self.board[fr][fc]
        captured_p = self.board[tr][tc]

        self.board[fr][fc] = "."
        self.board[tr][tc] = orig_p
        try:
            return not self.is_in_check(is_red=is_red)
        finally:
            self.board[fr][fc] = orig_p
            self.board[tr][tc] = captured_p

    def is_legal_move(self, uci_move: str) -> bool:
        """检验 UCI 着法 (如 'b2e2') 是否是当前方的合法着法"""
        uci_move = uci_move.strip().lower()
        if len(uci_move) != 4:
            return False
        legals = self.generate_legal_moves()
        return uci_move in legals

    def apply_move(self, uci_move: str) -> Tuple[bool, Optional[str]]:
        """
        在棋盘上执行 UCI 着法，返回 (是否成功, 被吃掉的棋子符号)
        """
        uci_move = uci_move.strip().lower()
        if not self.is_legal_move(uci_move):
            return False, None
        fc, fr = square_to_coord(uci_move[:2])
        tc, tr = square_to_coord(uci_move[2:])

        piece = self.board[fr][fc]
        captured = self.board[tr][tc]
        self.board[fr][fc] = "."
        self.board[tr][tc] = piece
        self.turn = "b" if self.turn == "w" else "w"
        return True, (captured if captured != "." else None)


class PikafishEngine:
    """Pikafish 象棋引擎控制器 (非阻塞队列读线程 + 有界超时 + 上下文管理器)"""

    def __init__(self, exe_path: Optional[str] = None):
        if not exe_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            exe_path = os.path.join(base_dir, "pikafish_bin", "Pikafish-Windows-x86-64-universal.exe")
        self.exe_path = exe_path
        self.proc: Optional[subprocess.Popen] = None
        self._line_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cmd_lock = threading.Lock()
        self._start_engine()

    def _reader_loop(self):
        proc = self.proc
        if not proc or not proc.stdout:
            return
        try:
            while not self._stop_event.is_set():
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue
                self._line_queue.put(line.strip())
        except Exception:
            pass
        finally:
            self._line_queue.put(None)

    def _start_engine(self):
        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"未找到 Pikafish 引擎可执行文件: {self.exe_path}")
        cwd = os.path.dirname(self.exe_path)
        self.proc = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            bufsize=1
        )
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True, name="PikafishReader")
        self._reader_thread.start()
        self._send("uci")
        self._wait_for("uciok", timeout=5.0)
        self._send("isready")
        self._wait_for("readyok", timeout=5.0)

    def _drain_queue(self):
        while not self._line_queue.empty():
            try:
                item = self._line_queue.get_nowait()
                if item is None:
                    self._line_queue.put(None)
                    break
            except queue.Empty:
                break

    def _send(self, cmd: str):
        if not self.proc or self.proc.poll() is not None or not self.proc.stdin:
            raise RuntimeError("Pikafish 引擎进程未运行")
        self.proc.stdin.write(f"{cmd}\n")
        self.proc.stdin.flush()

    def _wait_for(self, target: str, timeout: float = 3.0) -> List[str]:
        lines: List[str] = []
        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pikafish 响应超时 (期望 '{target}')")
            try:
                line_str = self._line_queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError(f"Pikafish 引擎意外退出 (code={self.proc.returncode})")
                continue
            if line_str is None:
                raise RuntimeError("Pikafish 标准输出流已关闭")
            lines.append(line_str)
            if target in line_str:
                return lines

    def get_best_move(self, fen: str, movetime_ms: int = 500) -> Tuple[str, Optional[str]]:
        """
        根据当前 FEN 局面，在限定时间内获取 Pikafish 最优走法
        返回: (best_move, ponder_move)
        """
        with self._cmd_lock:
            self._drain_queue()
            self._send("isready")
            self._wait_for("readyok", timeout=3.0)
            self._send(f"position fen {fen}")
            self._send(f"go movetime {int(movetime_ms)}")

            lines = self._wait_for("bestmove", timeout=max(5.0, (movetime_ms / 1000.0) + 3.0))
            for line in reversed(lines):
                if line.startswith("bestmove"):
                    parts = line.split()
                    best = parts[1] if len(parts) > 1 else ""
                    ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
                    return best, ponder
            return "", None

    def perft_moves(self, fen: str, timeout: float = 3.0) -> List[str]:
        """调用 Pikafish `go perft 1` 获取标准引擎给出的全部合法着法，用于规则微分校验。"""
        with self._cmd_lock:
            self._drain_queue()
            self._send("isready")
            self._wait_for("readyok", timeout=3.0)
            self._send(f"position fen {fen}")
            self._send("go perft 1")
            lines = self._wait_for("Nodes searched", timeout=timeout)
            moves: List[str] = []
            for line in lines:
                if ": " in line and not line.startswith("Nodes searched"):
                    mv = line.split(":", 1)[0].strip()
                    if len(mv) == 4:
                        moves.append(mv)
            return sorted(moves)

    def close(self):
        self._stop_event.set()
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                try:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
        except Exception:
            pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            if self._reader_thread and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)
            self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
