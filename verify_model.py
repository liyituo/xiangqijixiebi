import os
import glob
import safetensors.torch

files = sorted(glob.glob("/root/autodl-tmp/openvla-7b/*.safetensors"))
print("Found safetensors files:", files)
for f in files:
    try:
        t = safetensors.torch.load_file(f, device="cpu")
        print(f"SUCCESS loading {os.path.basename(f)}: {len(t)} keys")
    except Exception as e:
        print(f"FAIL loading {os.path.basename(f)}: {e}")
