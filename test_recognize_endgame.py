import cv2
import base64
import requests
import json
import os
import time
from env_loader import get_vlm_api_key

API_KEY = get_vlm_api_key("VLM_API_KEY", "DEEPSEEK_API_KEY")
API_URL = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL_NAME = "qwen3.8-flash"

img = cv2.imread("current_board_grid_preview.jpg")
ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = base64.b64encode(buf).decode("utf-8")

prompt = """这是一张带有中国象棋网格坐标标注(a0到i9)的棋盘俯拍图片。
坐标规则：
- 底线（红方底线）是 0 行：a0(红左车), b0, c0, d0(红士), e0(红帅), f0(红士), g0, h0, i0(红右车)
- 炮位是 2 行：b2, h2
- 兵位是 3 行：a3, c3, e3, g3, i3
- 楚河汉界在 4行与5行 之间
- 卒位是 6 行：a6, c6, e6, g6, i6
- 黑炮位是 7 行：b7, h7
- 黑方底线是 9 行：a9(黑左车), b9, c9, d9(黑士), e9(黑将), f9(黑士), g9, h9, i9(黑右车)

现在这是一个残局，棋盘上只有一部分棋子。
请仔细观察棋盘上实际存在的所有棋子，识别出棋子类型（大写红方: R车, N马, B相, A仕, K帅, C炮, P兵；小写黑方: r车, n马, b象, a士, k将, c炮, p卒）以及它们所在的网格坐标(如 a0, e0, e9, c3 等)。

请务必只输出严格的 JSON 格式：
{
  "pieces": [
    {"square": "e0", "piece": "K", "name": "红帅"},
    {"square": "e9", "piece": "k", "name": "黑将"}
  ],
  "turn": "r"
}
"""

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

t0 = time.time()
resp = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=30.0)
print("Status:", resp.status_code, "Elapsed:", time.time() - t0)
content = resp.json()["choices"][0]["message"].get("content", "")
print("Response:\n", content)

from vision_tools import _extract_vlm_json
parsed = _extract_vlm_json(content)
print("Parsed JSON:", parsed)
