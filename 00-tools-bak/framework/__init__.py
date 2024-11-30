#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2023/9/7 11:09
# @function:
import os
import torch
import platform

# 设置环境变量TOKENIZERS_PARALLELISM禁用或启用并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def detect_device():
    device = "mps" if getattr(torch, 'has_mps', False) \
        else "cuda" if torch.cuda.is_available() else "cpu"
    return device


has_gpu = torch.cuda.is_available()
has_mps = getattr(torch, 'has_mps', False)

print(f"Python Platform: {platform.platform()}")  # Python Platform: macOS-13.3.1-arm64-arm-64bit
print(f"PyTorch Version: {torch.__version__}")  # PyTsorch Version: 2.0.1

print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS is", "AVAILABLE" if has_mps else "NOT AVAILABLE")  # MPS is AVAILABLE

print(f"Target device is {detect_device()}")  # Target device is mps
