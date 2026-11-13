import io
import time
import base64
import requests
from PIL import Image

img = Image.new("RGB", (224, 224), color=(0, 200, 100))
buf = io.BytesIO()
img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

print("Testing /robot_vision_command...")
res = requests.post(
    "http://127.0.0.1:6006/robot_vision_command",
    json={
        "instruction": "抓取绿色零件",
        "image_base64": b64
    },
    timeout=20
)
print("Status:", res.status_code)
print("Response:", res.json())
