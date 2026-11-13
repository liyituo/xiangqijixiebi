import os
from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
files = api.list_repo_files("openvla/openvla-7b")

target_dir = "/root/autodl-tmp/openvla-7b"
os.makedirs(target_dir, exist_ok=True)

with open("/root/aria2_urls.txt", "w", encoding="utf-8") as f:
    for fname in files:
        if fname.startswith("."):
            continue
        local_path = os.path.join(target_dir, fname)
        aria2_file = local_path + ".aria2"
        # 只有文件存在且没有 .aria2 正在下载标记，且文件大小大于0，才算完成
        is_done = os.path.exists(local_path) and not os.path.exists(aria2_file) and os.path.getsize(local_path) > 0
        if not is_done:
            url = f"https://hf-mirror.com/openvla/openvla-7b/resolve/main/{fname}"
            f.write(f"{url}\n")
            f.write(f"  out={fname}\n")
            f.write(f"  dir={target_dir}\n")
            print(f"Adding to download: {fname}")

print("Updated /root/aria2_urls.txt successfully.")
