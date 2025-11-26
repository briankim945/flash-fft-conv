import time
import sys
import torch
from flashfftconv import FlashFFTConv, FlashFFTConvNoiFFT, convert_kernel_fft
from jax.lax import associative_scan
import jax.numpy as jnp
import numpy as np

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
        s = (u.size(-2), u.size(-1))
    l_h = u.size(-2)
    l_w = u.size(-1)
    u_f = torch.fft.fftn(u.to(torch.float32), dim=(-2,-1), s=s)
    k_f = torch.fft.fftn(k.to(torch.float32), dim=(-2,-1), s=s)
    u_f = u_f * k_f
    out = torch.fft.ifftn(u_f, dim=(-2,-1), s=s)
    return out.real.to(u.dtype)[..., :l_h, :l_w]

def ref_fft_conv_2d_12(u, k, s=None):
    if s is None:
        s = (u.size(-1), u.size(-2))
    l_h = u.size(-2)
    l_w = u.size(-1)
    u_f = torch.fft.fftn(u.to(torch.float32), dim=(-1,-2), s=s)
    k_f = torch.fft.fftn(k.to(torch.float32), dim=(-1,-2), s=s)
    u_f = u_f * k_f
    out = torch.fft.ifftn(u_f, dim=(-1,-2), s=s)
    return out.real.to(u.dtype)[..., :l_h, :l_w]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
B = 1
H = 3
L = 3

x = torch.randn(B, H, L, L, device=device).to(dtype) * 0.02
k = torch.randn(H, L, L, device=device) * 0.02

print(x)
print(k)
print()

x_clone = x.clone()
k_clone = k.clone()
out_2d = ref_fft_conv_2d(x_clone, k_clone)

x_clone_12 = x.clone()
k_clone_12 = k.clone()
out_2d_12 = ref_fft_conv_2d_12(x_clone_12, k_clone_12)

x = x.reshape(B, H, L * L)
k = k.reshape(H, L * L)

print(x)
print(k)
print()

out_1d = ref_fft_conv_1d(x, k)

print(out_1d)
print(out_2d)
print(out_2d_12)

sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seqlen = 2048 # 131072 # 2048 # 4096
dtype = torch.bfloat16

# # size of the FFT
# my_flashfftconv = FlashFFTConv(seqlen, dtype=dtype).to(device) # generally more stable!
# my_flashfftconv_noifft = FlashFFTConvNoiFFT(seqlen, dtype=dtype).to(device) # generally more stable!

print(torch.fft.fft(torch.randn(16, 64, 64)).shape)
print(torch.fft.fft(torch.randn(64, 64)).shape)
sys.exit()

# Testing Setup

img_list = [16, 32, 64, 128, 256]
kc_list = [10, 50, 100, 200]

time_dict_ifft = {l: {kc: [] for kc in kc_list} for l in img_list}
time_dict_no_ifft = {l: {kc: [] for kc in kc_list} for l in img_list}
time_dict_no_ifft_jax = {l: {kc: [] for kc in kc_list} for l in img_list}

for image_len in img_list:
    print("image_len:", image_len)
    for kernel_count in kc_list:
        print("\tkernel_count:", kernel_count)
        torch.cuda.empty_cache()
        # N = seqlen
        B = 16

        H = image_len
        L = image_len

        seqlen = shift_bit_length(max(image_len * 4, 256))

        # size of the FFT
        my_flashfftconv = FlashFFTConv(seqlen, dtype=dtype).to(device) # generally more stable!
        my_flashfftconv_noifft = FlashFFTConvNoiFFT(seqlen, dtype=dtype).to(device) # generally more stable!

        # print(L, image_L)

        x = torch.randn(B, H, L, device=device).to(dtype) * 0.02
        ks = [torch.randn(H, L, device=device) * 0.02 for _ in range(kernel_count)]
        mask = (torch.exp(-0.1 * torch.arange(0, L, device=device)))[:L]
        ks = [k * mask for k in ks]

        # 1. iFFT sequence

        x_clone = x.clone()
        ks_clones = [k.clone() for k in ks]

        time_start = time.perf_counter()
        out_flash = ref_fft_conv(x_clone, ks_clones[0])
        for k_clone in ks_clones[1:]:
            out_flash = ref_fft_conv(out_flash, k_clone)
        time_end = time.perf_counter()
        time_dict_ifft[image_len][kernel_count].append(time_end - time_start)

        # 2. No iFFT Sequence

        x_clone_f = x.clone()
        ks_clones_f = [convert_kernel_fft(k.clone(), seqlen) for k in ks]

        time_start = time.perf_counter()
        noifft_out_flash = my_flashfftconv_noifft(x_clone_f, ks_clones_f[0])
        for k_clone_f in ks_clones_f[1:]:
            noifft_out_flash = my_flashfftconv_noifft(noifft_out_flash, k_clone_f)
        time_end = time.perf_counter()
        time_dict_no_ifft[image_len][kernel_count].append(time_end - time_start)

        if not torch.allclose(out_flash, noifft_out_flash, atol=1e-2):
            print("out_flash, noifft_out_flash")

        # 3. No iFFT Parallel

        x_clone_f_scan = x.clone()
        ks_clones_f = jnp.array(
            [
                convert_kernel_fft(k.clone(), seqlen).cpu().numpy() for k in ks
            ]
        )

        time_start = time.perf_counter()
        scan_kernels = associative_scan(
            jnp.multiply,
            ks_clones_f
        )
        ks_to_scan = torch.tensor(np.array(scan_kernels[-1])).to(device)

        noifft_out_flash_scan = my_flashfftconv_noifft(x_clone_f_scan, ks_to_scan)
        time_end = time.perf_counter()
        time_dict_no_ifft_jax[image_len][kernel_count].append(time_end - time_start)

        if not torch.allclose(noifft_out_flash, noifft_out_flash_scan, atol=1e-2):
            print("out_fnoifft_out_flashlash, noifft_out_flash_scan")

print()
print()

for image_len in img_list:
    print("image_len:", image_len)
    for kernel_count in kc_list:
        print("\tkernel_count:", kernel_count)
        print(f"\t\tMean time sequential: {np.mean(time_dict_ifft[image_len][kernel_count]) / kernel_count}")
        print(f"\t\tMean time no iFFT: {np.mean(time_dict_no_ifft[image_len][kernel_count]) / kernel_count}")
        print(f"\t\tMean time no iFFT scan: {np.mean(time_dict_no_ifft_jax[image_len][kernel_count]) / kernel_count}")
