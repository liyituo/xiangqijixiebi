import os
import sys
import time
from huggingface_hub import snapshot_download

repo_id = "openvla/openvla-7b"
local_dir = "/root/autodl-tmp/openvla-7b"
endpoint = "https://hf-mirror.com"

print(f"[{time.strftime('%H:%M:%S')}] 启动从高速镜像源下载 {repo_id} 至 {local_dir} ...")
try:
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        endpoint=endpoint,
        max_workers=8
    )
    print(f"[{time.strftime('%H:%M:%S')}] 🎉 恭喜！{repo_id} 已完整下载至: {local_dir}")
except Exception as e:
    print(f"下载异常: {e}")
