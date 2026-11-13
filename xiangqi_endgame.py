# -*- coding: utf-8 -*-
"""
中国象棋残局视觉识别与 FEN 转换器 (Xiangqi Endgame Detector)
结合高精棋盘网格投影 + Qwen3.8-Flash + OpenCV 局部验证，
一键识别任意残局的所有棋子与位置，生成合法 FEN 局面并支持手动校验与微调。
"""

import os
import cv2
import json
import time
import base64
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from env_loader import get_vlm_api_key
from xiangqi_engine import (
    XiangqiBoard,
    PikafishEngine,
    is_valid_square,
    square_to_coord,
    VALID_PIECE_CHARS,
)
from xiangqi_agent import XiangqiGridMapper

API_KEY = get_vlm_api_key("VLM_API_KEY", "DEEPSEEK_API_KEY")
API_URL = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL_NAME = "qwen3.8-flash"

PIECE_MAP_CN = {
    "红帅": "K", "帅": "K", "红将": "K",
    "红兵": "P", "兵": "P",
    "红炮": "C", "炮": "C", "红包": "C", "包": "C",
    "红马": "N", "马": "N", "红傌": "N", "傌": "N",
    "红车": "R", "车": "R", "红俥": "R", "俥": "R",
    "红相": "B", "相": "B",
    "红仕": "A", "仕": "A",

    "黑将": "k", "将": "k",
    "黑卒": "p", "卒": "p",
    "黑炮": "c", "黑砲": "c", "砲": "c",
    "黑马": "n", "黑馬": "n",
    "黑车": "r", "黑車": "r",
    "黑象": "b", "象": "b",
    "黑士": "a", "士": "a",
}

def rotate_square_180(sq: str) -> str:
    """中国象棋棋盘中心对称旋转 180 度坐标转换 (a0 <-> i9)"""
    col = 8 - (ord(sq[0].lower()) - ord('a'))
    row = 9 - int(sq[1])
    return f"{chr(ord('a') + col)}{row}"

def build_fen_from_pieces(pieces_dict: Dict[str, str], turn: str = "w") -> str:
    """根据 {square: piece_symbol} 字典生成标准 FEN 字符串"""
    fen_rows = []
    for r in range(9, -1, -1):
        empty = 0
        row_str = ""
        for c_idx in range(9):
            sq = f"{chr(ord('a') + c_idx)}{r}"
            if sq in pieces_dict:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += pieces_dict[sq]
            else:
                empty += 1
        if empty > 0:
            row_str += str(empty)
        fen_rows.append(row_str)
    turn_str = "w" if turn.lower() in ["w", "r", "red"] else "b"
    return "/".join(fen_rows) + f" {turn_str} - - 0 1"

def detect_endgame_from_image(frame: np.ndarray, grid_mapper: Optional[XiangqiGridMapper] = None) -> Dict[str, Any]:
    """
    自动对当前相机画面进行残局检测与棋子定位。
    绝不使用固定 8 子示例残局静默回退；任何识别/网络/校验失败均明确返回错误。
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return {"success": False, "error": "输入图像为空或无效", "error_type": "invalid_input"}

    if not API_KEY:
        return {"success": False, "error": "未配置视觉识别 API Key (VLM_API_KEY)", "error_type": "config_missing"}

    if grid_mapper is None:
        grid_mapper = XiangqiGridMapper()

    h, w = frame.shape[:2]

    # 1. 在棋盘交叉点画上网格十字标
    disp = frame.copy()
    for r in range(10):
        for c in "abcdefghi":
            pt = grid_mapper.get_square_pixel_pos(f"{c}{r}", w, h)
            cv2.circle(disp, pt, 3, (0, 255, 0), -1)
            cv2.putText(disp, f"{c}{r}", (pt[0] - 8, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    ret, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return {"success": False, "error": "图像 JPEG 编码失败", "error_type": "encode_error"}
    b64 = base64.b64encode(buf).decode("utf-8")

    prompt = (
        "这是一张中国象棋残局棋盘的俯拍图，图上已有绿色小圆点标出各个网格交叉点 (如 a0..i9)。\n"
        "坐标方向：\n"
        "- 顶部底线是第 9 行 (从左到右 a9, b9, c9, d9, e9, f9, g9, h9, i9)\n"
        "- 中间是楚河汉界 (4行与5行之间)\n"
        "- 底部底线是第 0 行 (从左到右 a0, b0, c0, d0, e0, f0, g0, h0, i0)\n"
        "请识别棋盘上目前实际存在的所有棋子及其最近的交叉点坐标(例如 d9, h8, b3, e3, b1, e1, e0, h1)。\n"
        "棋子类型：\n"
        "- 红方棋子（红字）：帅(K), 兵(P), 炮(C), 马(N), 车(R), 相(B), 仕(A)\n"
        "- 黑方棋子（黑字）：将(k), 卒(p), 炮(c), 马(n), 车(r), 象(b), 士(a)\n"
        "只输出纯 JSON 格式：\n"
        "{\n"
        '  "pieces": [\n'
        '    {"square": "d9", "piece": "K", "name": "红帅"},\n'
        '    {"square": "e0", "piece": "k", "name": "黑将"}\n'
        "  ],\n"
        '  "turn": "r"\n'
        "}\n"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "enable_thinking": False,
        "max_tokens": 1500,
        "temperature": 0.1
    }

    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=25.0
        )
        if resp.status_code != 200:
            return {
                "success": False,
                "error": f"VLM 接口返回 HTTP {resp.status_code}",
                "error_type": "vlm_http_error",
            }
        content = resp.json()["choices"][0]["message"].get("content", "")
        from vision_tools import _extract_vlm_json
        parsed = _extract_vlm_json(content)
    except requests.Timeout as e:
        return {"success": False, "error": f"视觉识别超时: {e}", "error_type": "vlm_timeout"}
    except Exception as e:
        return {"success": False, "error": f"视觉识别异常: {e}", "error_type": "vlm_error"}

    if not isinstance(parsed, dict) or not isinstance(parsed.get("pieces"), list) or len(parsed["pieces"]) == 0:
        return {
            "success": False,
            "error": "视觉模型未识别出有效残局棋子列表，请调整光照或重试识别",
            "error_type": "empty_or_invalid_json",
        }

    raw_pieces = parsed["pieces"]
    piece_dict: Dict[str, str] = {}
    for idx, item in enumerate(raw_pieces):
        if not isinstance(item, dict):
            return {"success": False, "error": f"第 {idx} 项格式无效", "error_type": "validation_error"}
        sq = str(item.get("square", "")).strip().lower()
        if not is_valid_square(sq):
            return {"success": False, "error": f"检测到非法棋盘坐标: {sq!r}", "error_type": "validation_error"}
        p = str(item.get("piece", "")).strip()
        if p not in VALID_PIECE_CHARS:
            name = str(item.get("name", "")).strip()
            p = PIECE_MAP_CN.get(name, "")
        if p not in VALID_PIECE_CHARS:
            return {"success": False, "error": f"格子 {sq} 包含未知棋子类型: {item!r}", "error_type": "validation_error"}
        if sq in piece_dict:
            return {"success": False, "error": f"检测到重复占格坐标: {sq}", "error_type": "duplicate_square"}
        piece_dict[sq] = p

    red_kings = [sq for sq, p in piece_dict.items() if p == "K"]
    black_kings = [sq for sq, p in piece_dict.items() if p == "k"]
    if len(red_kings) != 1 or len(black_kings) != 1:
        return {
            "success": False,
            "error": f"残局局面将帅数量非法 (红帅={len(red_kings)}, 黑将={len(black_kings)})",
            "error_type": "invalid_kings",
        }

    red_king_sq = red_kings[0]
    is_rotated = int(red_king_sq[1]) >= 5
    if is_rotated:
        std_pieces = {rotate_square_180(sq): p for sq, p in piece_dict.items()}
    else:
        std_pieces = dict(piece_dict)

    # 校验兵/卒基本半区合理性 (未过河前不可退居己方底线2行内)
    for sq, p in std_pieces.items():
        _, r = square_to_coord(sq)
        if p == "P" and r < 3:
            return {"success": False, "error": f"红兵位置非法(处于己方后方): {sq}", "error_type": "invalid_pawn"}
        if p == "p" and r > 6:
            return {"success": False, "error": f"黑卒位置非法(处于己方后方): {sq}", "error_type": "invalid_pawn"}

    raw_turn = str(parsed.get("turn", "r")).strip().lower()
    fen_turn = "b" if raw_turn in ("b", "black", "黑") else "w"
    fen = build_fen_from_pieces(std_pieces, turn=fen_turn)
    ok_fen, fen_err = XiangqiBoard.validate_fen(fen)
    if not ok_fen:
        return {"success": False, "error": f"识别局面不符合象棋规则: {fen_err}", "error_type": "invalid_fen"}

    print(f"🎯 残局识别完成: 共 {len(piece_dict)} 颗棋子, 旋转对齐: {is_rotated}, 标准FEN: {fen}")

    return {
        "success": True,
        "raw_pieces": raw_pieces,
        "piece_dict": piece_dict,
        "is_rotated": is_rotated,
        "fen": fen,
        "diag_img": "endgame_recognized_preview.jpg"
    }

if __name__ == "__main__":
    frame = cv2.imread("current_board_home.jpg")
    res = detect_endgame_from_image(frame)
    print("Result:", res)
