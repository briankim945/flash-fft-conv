import torch
from flashfftconv import FlashFFTConv

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

# B is batch size, H is model dimension, L is sequence length
B = 16
H = 192
# input can be smaller than FFT size, but needs to be divisible by 2
L = 4096

# the input, B H L
# x = torch.randn(B, H, L, dtype=torch.bfloat16).to(device) # same type as the input
# k = torch.randn(H, L, dtype=torch.float32).to(device) * 0.02 # kernel needs to be fp32 for now

# x[:, :, seqlen // 2 :] = 0.
# k[:, seqlen // 2 :] = 0.
# mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen]
# k = k * mask

N = seqlen
x = torch.randn(B, H, N // 2, device=device).to(dtype) * 0.02
k = torch.randn(H, N // 2, device=device) * 0.02
mask = mask = (torch.exp(-0.1 * torch.arange(0, seqlen, device=device)))[:seqlen // 2]
k = k * mask

x_clone = x.clone()
k_clone = k.clone()

out_flash = my_flashfftconv(x, k)

out_ref = ref_fft_conv(x_clone, k_clone)

print(out_flash.shape)
print(out_ref.shape)
print(torch.allclose(out_flash, out_ref, atol=1e-2))

abs_error = torch.abs(out_flash - out_ref)

print(f"Abs Error Mean: {abs_error.mean():.3E}")
print(f"Abs Error Std Dev: {abs_error.std():.3E}")
print(f"Abs Error Total: {abs_error.sum():.3E}")