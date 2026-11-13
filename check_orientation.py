import requests
import base64
from env_loader import get_vlm_api_key

KEY = get_vlm_api_key("DIRECT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
URL = "https://api.deepseek.com/chat/completions"

with open("current_table_snapshot.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

prompt = """请仔细观察这张图片的方向与物体布局：
1. 画面中可以看到机械臂基座、黄色急停按钮盒、或者桌子边缘吗？分别在画面的什么方位（上/下/左/右）？
2. 画面中手绘的矩形框和4个黑点：
   - 图像左上方的黑点，旁边写的是什么字或者标号？
   - 图像右上方的黑点，旁边写的是什么？
   - 图像右下方的黑点，旁边写的是什么？
   - 图像左下方的黑点，旁边写的是什么？
请详细描述每个角点和基座的方向关系。"""

payload = {
    "model": "deepseek-flash",
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}
    ],
    "thinking": {"type": "disabled"},
    "temperature": 0.1
}

r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, json=payload, timeout=20)
print(r.json()["choices"][0]["message"].get("content"))
