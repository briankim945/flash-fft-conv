import torch 
import time
from torch import nn
from einops import rearrange
from flashfftconv import FlashDepthWiseConv1d
import pytest
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dtype = (torch.bfloat16, torch.bfloat16)

b = 1
h = 768
l = 1024
k = 3

in_dtype = dtype[0]
w_dtype = dtype[1]

padding =  (k -1)//2
            
x = torch.randn([b, h, l], device=device, dtype=in_dtype)
x_float = x.clone().to(torch.float32)

conv1d_torch = nn.Conv1d(
    in_channels = h,
    out_channels = h,
    kernel_size = k,
    groups = h,
    padding = padding,
    dtype = w_dtype,
    device = device
)

conv1d_cuda = FlashDepthWiseConv1d(
    channels = h,
    kernel_size=k,
    padding=padding,
    weights=conv1d_torch.weight,
    bias=conv1d_torch.bias,
    dtype = w_dtype,
    device = device
)

with torch.autocast(device_type='cuda', dtype=in_dtype):
    y_torch = conv1d_torch(x)
y_cuda = conv1d_cuda(x)
with torch.autocast(device_type='cuda', dtype=in_dtype):
    y_cuda_autocast = conv1d_cuda(x_float).to(in_dtype)
with torch.autocast(device_type='cuda', dtype=in_dtype):
    y_cuda_manual_cast = conv1d_cuda(x_float.to(in_dtype))

print(torch.allclose(y_torch, y_cuda, atol=1e-1))
print(torch.allclose(y_torch, y_cuda_autocast, atol=1e-1))
print(torch.allclose(y_torch, y_cuda_manual_cast, atol=1e-1))
