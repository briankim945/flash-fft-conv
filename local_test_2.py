import torch
from flashfftconv import FlashFFTConv, FlashFFTConvNoiFFT, convert_kernel_fft
from jax.lax import associative_scan
import jax.numpy as jnp
import numpy as np

def ref_fft_conv(u, k, n=None):
    if n is None:
        n = u.size(-1)
    l = u.size(-1)
    u_f = torch.fft.fft(u.to(torch.float32), n=n)
    k_f = torch.fft.fft(k.to(torch.float32), n=n)
    u_f = u_f * k_f
    out = torch.fft.ifft(u_f, n=n)
    return out.real.to(u.dtype)[..., :l]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seqlen = 4096
dtype = torch.bfloat16

# size of the FFT
my_flashfftconv = FlashFFTConv(seqlen, dtype=dtype).to(device) # generally more stable!
my_flashfftconv_noifft = FlashFFTConvNoiFFT(seqlen, dtype=dtype).to(device) # generally more stable!

# # B is batch size, H is model dimension, L is sequence length
# B = 16
# H = 192
# # input can be smaller than FFT size, but needs to be divisible by 2
# L = N // 2 # divide in 2 for padding

# N = seqlen
# x = torch.randn(B, H, L, device=device).to(dtype) * 0.02
# k = torch.randn(H, L, device=device) * 0.02
# mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen // 2]
# k = k * mask

# x_clone = x.clone()
# k_clone = k.clone()

# out_flash = my_flashfftconv(x, k)

# out_ref = ref_fft_conv(x_clone, k_clone)

# print(out_flash.shape)
# print(out_ref.shape)
# print(torch.allclose(out_flash, out_ref, atol=1e-2))

# abs_error = torch.abs(out_flash - out_ref)

# print(f"Abs Error Mean: {abs_error.mean():.3E}")
# print(f"Abs Error Std Dev: {abs_error.std():.3E}")
# print(f"Abs Error Total: {abs_error.sum():.3E}")
# print()
# print()


# Testing Setup

N = seqlen
# B is batch size, H is model dimension, L is sequence length
B = 16
H = 192
# input can be smaller than FFT size, but needs to be divisible by 2
L = N // 2 # divide in 2 for padding

x = torch.randn(B, H, L, device=device).to(dtype) * 0.02
k1 = torch.randn(H, L, device=device) * 0.02
k2 = torch.randn(H, L, device=device) * 0.02
k3 = torch.randn(H, L, device=device) * 0.02
k4 = torch.randn(H, L, device=device) * 0.02
mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen // 2]
k1 = k1 * mask
k2 = k2 * mask
k3 = k3 * mask
k4 = k4 * mask

# 1. iFFT sequence

x_clone = x.clone()
k1_clone = k1.clone()
k2_clone = k2.clone()
k3_clone = k3.clone()
k4_clone = k4.clone()

out_flash1 = ref_fft_conv(x_clone, k1_clone)
out_flash2 = ref_fft_conv(out_flash1, k2_clone)
out_flash3 = ref_fft_conv(out_flash2, k3_clone)
out_flash4 = ref_fft_conv(out_flash3, k4_clone)

# 2. No iFFT Sequence

x_clone_f = x.clone()
k1_clone_f = convert_kernel_fft(k1.clone(), seqlen)
k2_clone_f = convert_kernel_fft(k2.clone(), seqlen)
k3_clone_f = convert_kernel_fft(k3.clone(), seqlen)
k4_clone_f = convert_kernel_fft(k4.clone(), seqlen)

noifft_out_flash1 = my_flashfftconv_noifft(x_clone_f, k1_clone_f)
noifft_out_flash2 = my_flashfftconv_noifft(noifft_out_flash1, k2_clone_f)
noifft_out_flash3 = my_flashfftconv_noifft(noifft_out_flash2, k3_clone_f)
noifft_out_flash4 = my_flashfftconv_noifft(noifft_out_flash3, k4_clone_f)

print(out_flash4.shape)
print(noifft_out_flash4.shape)
print(torch.allclose(out_flash4, noifft_out_flash4, atol=1e-2))

abs_error = torch.abs(out_flash4 - noifft_out_flash4)

print(f"Abs Error Mean: {abs_error.mean():.3E}")
print(f"Abs Error Std Dev: {abs_error.std():.3E}")
print(f"Abs Error Total: {abs_error.sum():.3E}")
print()

# 3. No iFFT Parallel
# TODO

print(associative_scan(jnp.add, jnp.arange(0, 4)))

x_clone_f_scan = x.clone()
k1_clone_f_scan = convert_kernel_fft(k1.clone(), seqlen).cpu().numpy()
k2_clone_f_scan = convert_kernel_fft(k2.clone(), seqlen).cpu().numpy()
k3_clone_f_scan = convert_kernel_fft(k3.clone(), seqlen).cpu().numpy()
k4_clone_f_scan = convert_kernel_fft(k4.clone(), seqlen).cpu().numpy()

# scan_kernels = associative_scan(
#     jnp.multiply,
#     jnp.array([k1_clone_f_scan, k2_clone_f_scan, k3_clone_f_scan, k4_clone_f_scan])
# )
# # print(len(scan_kernels))
# # print([scan_item.shape for scan_item in scan_kernels])
# print(scan_kernels.shape)

scan_kernels = associative_scan(
    torch.multiply,
    torch.tensor([k1_clone_f_scan, k2_clone_f_scan, k3_clone_f_scan, k4_clone_f_scan]),
    axis=0
)
# print(len(scan_kernels))
# print([scan_item.shape for scan_item in scan_kernels])
print(scan_kernels.shape)

# ks_to_scan_4 = torch.tensor(np.array(scan_kernels[-1])).to(device)
ks_to_scan_4 = scan_kernels[-1].to(device)

noifft_out_flash4_scan = my_flashfftconv_noifft(x_clone_f_scan, ks_to_scan_4)

print(noifft_out_flash4.shape)
print(noifft_out_flash4_scan.shape)
print(torch.allclose(noifft_out_flash4, noifft_out_flash4_scan, atol=1e-2))

abs_error = torch.abs(noifft_out_flash4 - noifft_out_flash4_scan)

print(f"Abs Error Mean: {abs_error.mean():.3E}")
print(f"Abs Error Std Dev: {abs_error.std():.3E}")
print(f"Abs Error Total: {abs_error.sum():.3E}")
print()
