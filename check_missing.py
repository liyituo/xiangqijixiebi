import os
from huggingface_hub import HfApi

api = HfApi(endpoint="https://hf-mirror.com")
files = api.list_repo_files("openvla/openvla-7b")

target_dir = "/root/autodl-tmp/openvla-7b"
urls = []
for f in files:
    if f.startswith("."):
        continue
    local_path = os.path.join(target_dir, f)
    # Check if already fully downloaded
    if not os.path.exists(local_path):
        url = f"https://hf-mirror.com/openvla/openvla-7b/resolve/main/{f}"
        urls.append((f, url))

print(f"Total files in repo: {len(files)}, missing files: {len(urls)}")
for f, u in urls:
    print(f"Missing: {f} -> {u}")
