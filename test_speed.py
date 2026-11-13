import os
import time
from huggingface_hub import hf_hub_download

print("Testing download with hf_transfer...")
t0 = time.time()
p = hf_hub_download(
    repo_id="openvla/openvla-7b",
    filename="generation_config.json",
    local_dir="/root/autodl-tmp/test_download",
    resume_download=True
)
t1 = time.time()
print(f"Downloaded {p} in {t1 - t0:.2f}s")
