# _*_ coding : UTF-8 _*_
# @Time : 2026/5/12 17:05
# @Author : Yif Wang
# @file : ddp
import torch
import torch.distributed as dist

class ddp(torch.nn.Module):
    def __init__(self):
        super().__init__()
