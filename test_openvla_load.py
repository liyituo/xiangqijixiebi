import os
import torch
from transformers import AutoConfig, AutoProcessor

model_path = "/root/autodl-tmp/openvla-7b"
print("Loading config...")
try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("Config OK:", type(config))
except Exception as e:
    print("Config error:", e)

print("Loading processor...")
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Processor OK:", type(processor))
except Exception as e:
    print("Processor error:", e)

print("Testing model weight loading...")
try:
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("Model load OK!")
except Exception as e:
    print("Model load error:", e)
