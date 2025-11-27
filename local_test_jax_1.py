import time
import sys
import jax
import jax.numpy as jnp
from jax import random
from jax.lax import associative_scan
from flashfftconv import manual_flash_fft_conv_2d, manual_flash_fft_conv_2d_noifft

def ref_fft_conv_2d(u: jnp.ndarray, k: jnp.ndarray, s: tuple = None) -> jnp.ndarray:
    """Reference 2D FFT convolution"""
    if s is None:
        s = (u.shape[-2], u.shape[-1])
    l_h = u.shape[-2]
    l_w = u.shape[-1]
    u_f = jnp.fft.fftn(u.astype(jnp.float32), axes=(-2, -1), s=s)
    k_f = jnp.fft.fftn(k.astype(jnp.float32), axes=(-2, -1), s=s)
    u_f = u_f * k_f
    out = jnp.fft.ifftn(u_f, axes=(-2, -1), s=s)
    return out.real.astype(u.dtype)[..., :l_h, :l_w]

# JAX device selection (will use GPU if available)
device = jax.devices()[0]
print(f"Using device: {device}")

# Note: JAX doesn't have bfloat16 as a direct dtype in the same way,
# but it's supported on TPUs and some GPUs
dtype = jnp.bfloat16
key = random.key(0)

seqlen = 4096

# # size of the FFT
# my_flashfftconv = FlashFFTConv(seqlen, dtype=dtype).to(device) # generally more stable!

# B is batch size, H is model dimension, L is sequence length
B = 16
H = 192
# input can be smaller than FFT size, but needs to be divisible by 2
L = 64 #4096
kernel_T = 10 # Number of kernels

x = random.normal(key, (B, H, L, L)) * 0.02
ks = [random.normal(key, (H, L, L)) * 0.02 for _ in range(kernel_T)]
mask = (jnp.exp(-0.1 * jnp.arange(0, L * L)))[:L*L].reshape(L,L)
ks = [k * mask for k in ks]

x_clone = x.clone()
ks_clones = [k.clone() for k in ks]

# out_flash = my_flashfftconv(x, k)

# out_ref = ref_fft_conv(x_clone, k_clone)

### STANDARD FFT

time_start = time.perf_counter()
out_ref = ref_fft_conv_2d(x_clone, ks_clones[0])
for k_clone in ks_clones[1:]:
    out_ref = ref_fft_conv_2d(out_ref, k_clone)
time_end = time.perf_counter()
out_ref_time = time_end - time_start
# time_dict_ifft[image_len][kernel_count].append(time_end - time_start)

### FFT FLASH NO iFFT SEQUENTIAL

x_clone_flash = x.clone()
ks_clones_flash = [k.clone() for k in ks]
s = (x_clone_flash.shape[-2], x_clone_flash.shape[-1])

time_start = time.perf_counter()
ks_clones_flash_fft = [jnp.fft.fftn(k, axes=(-2, -1), s=s) for k in ks_clones_flash]
k_cum_flash = ks_clones_flash[0] * ks_clones_flash[1]
for k_clones_flash in ks_clones_flash[2:]:
    k_cum_flash = k_cum_flash * k_clones_flash
out_flash_seq = manual_flash_fft_conv_2d_noifft(x_clone_flash, k_cum_flash, seqlen)
time_end = time.perf_counter()
out_flash_time = time_end - time_start

### FFT FLASH NO iFFT SCAN

x_clone_flash_scan = x.clone()
ks_clones_flash_scan = [k.clone() for k in ks]
s = (x_clone_flash_scan.shape[-2], x_clone_flash_scan.shape[-1])

time_start = time.perf_counter()
ks_clones_flash_fft = jnp.array([jnp.fft.fftn(k, axes=(-2, -1), s=s) for k in ks_clones_flash_scan])
scan_kernels = associative_scan(jnp.multiply, ks_clones_flash_fft)
out_flash_seq_scan = manual_flash_fft_conv_2d_noifft(x_clone_flash_scan, scan_kernels[-1], seqlen)
time_end = time.perf_counter()
out_flash_scan_time = time_end - time_start

print(out_ref.shape)
print(out_flash_seq.shape)
print(out_flash_seq_scan.shape)
print()

print(jnp.allclose(out_ref, out_flash_seq, atol=1e-2))
print(jnp.allclose(out_flash_seq, out_flash_seq_scan, atol=1e-2))
print()

print(out_ref_time)
print(out_flash_time)
print(out_flash_scan_time)

# abs_error = jnp.abs(out_flash - out_ref)

# print(f"Abs Error Mean: {abs_error.mean():.3E}")
# print(f"Abs Error Std Dev: {abs_error.std():.3E}")
# print(f"Abs Error Total: {abs_error.sum():.3E}")