#!/usr/bin/env python3
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
print("clés du checkpoint:", list(ckpt.keys()))
print()
print("args:")
for k, v in sorted(ckpt.get("args", {}).items()):
    print(f"  {k} = {v}")
