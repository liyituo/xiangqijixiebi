import os
import sys

sys.path.insert(0, "/root/autodl-tmp/openvla-7b")
from processing_prismatic import PrismaticProcessor

try:
    p = PrismaticProcessor.from_pretrained("/root/autodl-tmp/openvla-7b", trust_remote_code=True)
    print("PrismaticProcessor SUCCESS:", type(p))
except Exception as e:
    print("PrismaticProcessor ERROR:", e)
