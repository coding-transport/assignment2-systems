# _*_ coding : UTF-8 _*_
# @Time : 2026/5/12 17:01
# @Author : Yif Wang
# @file : interface
from cs336_systems.attention.pytorch_flash import MyFlashAttnAutogradFunctionClass
import torch

def get_flashattention_autograd_function_pytorch():
    return MyFlashAttnAutogradFunctionClass