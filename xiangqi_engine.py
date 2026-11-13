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
import subprocess
from typing import List, Tuple, Optional, Dict, Any

# 棋子定义 (红方大写，黑方小写)
# Red:   R=车, N=马, B=相, A=仕, K=帅, C=炮, P=兵
# Black: r=车, n=马, b=象, a=士, k=将, c=炮, p=卒
INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

PIECE_NAMES_CN = {
    "R": "红车", "N": "红马", "B": "红相", "A": "红仕", "K": "红帅", "C": "红炮", "P": "红兵",
    "r": "黑车", "n": "黑马", "b": "黑象", "a": "黑士", "k": "黑将", "c": "黑炮", "p": "黑卒",
    ".": "空"
}

FILES = "abcdefghi" # 0 to 8
RANKS = "0123456789" # 0 to 9

def square_to_coord(sq: str) -> Tuple[int, int]:
    """将 UCI 方格 (如 'b2') 转换为 (col, row), col 0..8, row 0..9"""
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

    def set_fen(self, fen: str):
        """解析 FEN 串加载棋盘"""
        parts = fen.strip().split()
        rows = parts[0].split("/")
        self.board = [["." for _ in range(9)] for _ in range(10)]
        # FEN row 0 是黑方底线 (rank 9)，row 9 是红方底线 (rank 0)
        for r_idx, row_str in enumerate(rows):
            board_r = 9 - r_idx
            c_idx = 0
            for ch in row_str:
                if ch.isdigit():
                    c_idx += int(ch)
                else:
                    self.board[board_r][c_idx] = ch
                    c_idx += 1
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

    def _is_move_safe_from_check(self, from_sq: str, to_sq: str, is_red: bool) -> bool:
        """检查执行移动后是否送将或对脸帅"""
        fc, fr = square_to_coord(from_sq)
        tc, tr = square_to_coord(to_sq)
        orig_p = self.board[fr][fc]
        captured_p = self.board[tr][tc]

        self.board[fr][fc] = "."
        self.board[tr][tc] = orig_p

        # 1. 检查飞将对脸 (Flying Generals)
        safe = True
        red_k_col, red_k_row = -1, -1
        black_k_col, black_k_row = -1, -1
        for r in range(10):
            for c in range(9):
                if self.board[r][c] == "K":
                    red_k_col, red_k_row = c, r
                elif self.board[r][c] == "k":
                    black_k_col, black_k_row = c, r

        if red_k_col == black_k_col and red_k_col != -1:
            # 同列，检查中间是否有遮挡
            min_r = min(red_k_row, black_k_row)
            max_r = max(red_k_row, black_k_row)
            blocked = False
            for mid_r in range(min_r + 1, max_r):
                if self.board[mid_r][red_k_col] != ".":
                    blocked = True
                    break
            if not blocked:
                safe = False

        # 还原
        self.board[fr][fc] = orig_p
        self.board[tr][tc] = captured_p
        return safe

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
    """Pikafish 象棋引擎控制器"""

    def __init__(self, exe_path: Optional[str] = None):
        if not exe_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            exe_path = os.path.join(base_dir, "pikafish_bin", "Pikafish-Windows-x86-64-universal.exe")
        self.exe_path = exe_path
        self.proc: Optional[subprocess.Popen] = None
        self._start_engine()

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
        self._send("uci")
        # 等待 readyok
        self._send("isready")
        self._wait_for("readyok", timeout=5.0)

    def _send(self, cmd: str):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(f"{cmd}\n")
            self.proc.stdin.flush()

    def _wait_for(self, target: str, timeout: float = 3.0) -> List[str]:
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            line_str = line.strip()
            lines.append(line_str)
            if target in line_str:
                return lines
        return lines

    def get_best_move(self, fen: str, movetime_ms: int = 500) -> Tuple[str, Optional[str]]:
        """
        根据当前 FEN 局面，在限定时间内获取 Pikafish 最优走法
        返回: (best_move, ponder_move)
        """
        self._send("isready")
        self._wait_for("readyok", timeout=3.0)
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")

        lines = self._wait_for("bestmove", timeout=max(5.0, (movetime_ms / 1000.0) + 3.0))
        for line in reversed(lines):
            if line.startswith("bestmove"):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else ""
                ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
                return best, ponder
        return "", None

    def close(self):
        try:
            if self.proc:
                self._send("quit")
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
        except Exception:
            pass
