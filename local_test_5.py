import time
import sys
import torch
from flashfftconv import FlashFFTConv, FlashFFTConvNoiFFT, convert_kernel_fft
from flashfftconv.conv import fft_matrix, compute_twiddle_factors_fft, ifft_matrix,\
    compute_twiddle_factors_ifft, monarch_outer_dft, monarch_outer_idft
from jax.lax import associative_scan
import jax.numpy as jnp
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dtype = torch.bfloat16

BLOCK_DIM_X = 32
BLOCK_DIM_Y = 1
N = 256
MATMUL_WARP_WIDTH = 1
RECOMPUTE = False
B_TILE_SIZE = 1
H_TILE_SIZE = 1

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

def ref_fft_conv_1d(u, k, n=None):
    if n is None:
        n = u.size(-1)
    l = u.size(-1)
    u_f = torch.fft.fft(u.to(torch.float32), n=n)
    k_f = torch.fft.fft(k.to(torch.float32), n=n)
    u_f = u_f * k_f
    out = torch.fft.ifft(u_f, n=n)
    return out.real.to(u.dtype)[..., :l]

def ref_fft_conv_2d(u, k, s=None):
    if s is None:
        s = (u.size(-1), u.size(-2))
    l_h = u.size(-2)
    l_w = u.size(-1)
    u_f = torch.fft.fftn(u.to(torch.float32), dim=(-1,-2), s=s)
    k_f = torch.fft.fftn(k.to(torch.float32), dim=(-1,-2), s=s)
    u_f = u_f * k_f
    out = torch.fft.ifftn(u_f, dim=(-1,-2), s=s)
    return out.real.to(u.dtype)[..., :l_h, :l_w]

def manual_flash_fft_conv(u, k, seqlen):
    # k_f = convert_kernel_fft(k, seqlen)
    if seqlen in [512, 2048]:
        k_f = torch.fft.rfft(k, n=seqlen)
    else:
        k_f = torch.fft.fft(k, n=seqlen)
    # print("u:", u.shape)
    # print("k_f:", k_f.shape)

    N = seqlen
    sqrt_N = int(math.sqrt(seqlen))
    # self.N = N
    # self.sqrt_N = sqrt_N

    # view_as_real needed for cuda due to complex matmul
    # f_sqrt_N_fft = torch.view_as_real(fft_matrix(sqrt_N)).to(dtype).to(device)
    # f_sqrt_N_ifft = torch.view_as_real(ifft_matrix(sqrt_N)).to(dtype).to(device)

    # twiddle_factors_fft = torch.view_as_real(compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).to(dtype).to(device)
    # twiddle_factors_ifft = torch.view_as_real(compute_twiddle_factors_ifft(sqrt_N, sqrt_N)).to(dtype).to(device)

    f_sqrt_N_fft = fft_matrix(sqrt_N).to(dtype).to(device)
    f_sqrt_N_ifft = ifft_matrix(sqrt_N).to(dtype).to(device)

    twiddle_factors_fft = (compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).to(dtype).to(device)
    twiddle_factors_ifft = (compute_twiddle_factors_ifft(sqrt_N, sqrt_N)).to(dtype).to(device)

    # print("f_sqrt_N_fft:", f_sqrt_N_fft.shape)
    # print("f_sqrt_N_ifft:", f_sqrt_N_ifft.shape)
    # print("twiddle_factors_fft:", twiddle_factors_fft.shape)
    # print("twiddle_factors_ifft:", twiddle_factors_ifft.shape)

    # self.register_buffer('f_sqrt_N_fft', f_sqrt_N_fft)
    # self.register_buffer('f_sqrt_N_ifft', f_sqrt_N_ifft)
    # self.register_buffer('twiddle_factors_fft', twiddle_factors_fft)
    # self.register_buffer('twiddle_factors_ifft', twiddle_factors_ifft)

    # k_f_permuted = torch.view_as_real(k_f.reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N)).to(dtype).contiguous()
    k_f_permuted = k_f.reshape(H, sqrt_N, sqrt_N).transpose(-1, -2).reshape(H, N).to(dtype).contiguous()
    k_f_permuted = k_f_permuted.to(device)
    # print("k_f_permuted:", k_f_permuted.shape)

    output = torch.zeros(*u.shape, dtype=dtype, device=device)

    u_pad = torch.nn.functional.pad(u, (0, seqlen - u.shape[-1])).reshape(u.shape[0], u.shape[1], sqrt_N, sqrt_N)
    # print("u_pad:", u_pad.shape)

    for h_tile_id in range(H_TILE_SIZE):
        k_f_cur = k_f_permuted[h_tile_id].reshape(sqrt_N, sqrt_N)
        for b_tile_id in range(B_TILE_SIZE):
            # print(torch.matmul(f_sqrt_N_fft.T, u_pad[b_tile_id][h_tile_id]).shape)
            fft_2_step = torch.matmul(torch.matmul(f_sqrt_N_fft.T, u_pad[b_tile_id][h_tile_id]) * twiddle_factors_fft, f_sqrt_N_fft)
            # print(fft_2_step.shape)
            elemwise_mul = fft_2_step * k_f_cur
            ifft_2_step = torch.matmul(torch.matmul(elemwise_mul, f_sqrt_N_ifft).T * twiddle_factors_ifft, f_sqrt_N_ifft)
            # print(output.shape, ifft_2_step.shape)
            output[b_tile_id][h_tile_id] = ifft_2_step.T.reshape(seqlen)[:u.shape[-1]]
    return output

### VARIABLES

seqlen = 4096
# B is batch size, H is model dimension, L is sequence length
B = 16
H = 192
# input can be smaller than FFT size, but needs to be divisible by 2
L = 4096
N = seqlen
x = torch.randn(B, H, N // 2, device=device).to(dtype) * 0.02
k = torch.randn(H, N // 2, device=device) * 0.02
mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen // 2]
k = k * mask

x_clone = x.clone()
k_clone = k.clone()


### REFERENCE RESULT

out_ref = ref_fft_conv_1d(x_clone, k_clone)
print("out_ref:", out_ref.shape)
print()


### MANUAL FLASH

out_flash = manual_flash_fft_conv(x, k, seqlen)
print("out_flash:", out_flash.shape)
print()

# print(out_ref.shape)
print(torch.allclose(out_flash, out_ref, atol=1e-2))

abs_error = torch.abs(out_flash - out_ref)

print(f"Abs Error Mean: {abs_error.mean():.3E}")
print(f"Abs Error Std Dev: {abs_error.std():.3E}")
print(f"Abs Error Total: {abs_error.sum():.3E}")
