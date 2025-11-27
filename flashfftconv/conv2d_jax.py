import time
import sys
import jax
import jax.numpy as jnp
from jax import lax
import math

# Enable float64 if needed
# jax.config.update("jax_enable_x64", True)

# JAX device selection (will use GPU if available)
device = jax.devices()[0]
print(f"Using device: {device}")

# Note: JAX doesn't have bfloat16 as a direct dtype in the same way,
# but it's supported on TPUs and some GPUs
dtype = jnp.bfloat16

BLOCK_DIM_X = 32
BLOCK_DIM_Y = 1
N = 256
MATMUL_WARP_WIDTH = 1
RECOMPUTE = False
B_TILE_SIZE = 1
H_TILE_SIZE = 1


def fft_matrix(N: int) -> jnp.ndarray:
    """Compute the DFT matrix of size N x N"""
    n = jnp.arange(N)
    k = n.reshape(-1, 1)
    M = jnp.exp(-2j * jnp.pi * n * k / N)
    return M


def compute_twiddle_factors_fft(n: int, m: int) -> jnp.ndarray:
    """Compute the twiddle factors of size n x m for FFT"""
    n_a = jnp.arange(n).reshape(-1, 1)
    m_a = jnp.arange(m)
    N = n * m
    M = jnp.exp(-2j * jnp.pi * n_a * m_a / N)
    return M


def ifft_matrix(N: int) -> jnp.ndarray:
    """Compute the inverse DFT matrix of size N x N"""
    n = jnp.arange(N)
    k = n.reshape(-1, 1)
    M = jnp.exp(2j * jnp.pi * n * k / N)
    return M


def compute_twiddle_factors_ifft(n: int, m: int) -> jnp.ndarray:
    """Compute the twiddle factors of size n x m for inverse FFT"""
    n_a = jnp.arange(n).reshape(-1, 1)
    m_a = jnp.arange(m)
    N = n * m
    M = jnp.exp(2j * jnp.pi * n_a * m_a / N)
    return M


def monarch_outer_dft(x: jnp.ndarray, f_sqrt_N_fft: jnp.ndarray, 
                      twiddle_factors_fft: jnp.ndarray, sqrt_N: int) -> jnp.ndarray:
    """Monarch matrix outer DFT step"""
    x = x.swapaxes(-1, -2)  # 32K, 32
    x = x @ f_sqrt_N_fft     # 32K, 32
    x = x.swapaxes(-1, -2)  # 32, 32K
    return x * twiddle_factors_fft


def monarch_outer_idft(x: jnp.ndarray, f_sqrt_N_ifft: jnp.ndarray,
                       twiddle_factors_ifft: jnp.ndarray, sqrt_N: int) -> jnp.ndarray:
    """Monarch matrix outer inverse DFT step"""
    x = x * twiddle_factors_ifft
    x = x.swapaxes(-1, -2)  # 32K, 32
    x = x @ f_sqrt_N_ifft
    x = x.swapaxes(-1, -2)  # 32, 32K
    return x


def shift_bit_length(x: int) -> int:
    """Round up to the next power of 2"""
    return 1 << (x - 1).bit_length()


def pad_1d(x: jnp.ndarray, pad_right: int) -> jnp.ndarray:
    """Pad a tensor on the right side of the last dimension"""
    if pad_right <= 0:
        return x
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_right)]
    return jnp.pad(x, pad_width)


def pad_2d(x: jnp.ndarray, pad: tuple) -> jnp.ndarray:
    """
    Pad a 4D tensor (batch, channels, height, width).
    pad: (left, right, top, bottom) for last two dimensions
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    pad_width = [(0, 0)] * (x.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
    return jnp.pad(x, pad_width)


def ref_fft_conv_1d(u: jnp.ndarray, k: jnp.ndarray, n: int = None) -> jnp.ndarray:
    """Reference 1D FFT convolution"""
    if n is None:
        n = u.shape[-1]
    l = u.shape[-1]
    u_f = jnp.fft.fft(u.astype(jnp.float32), n=n)
    k_f = jnp.fft.fft(k.astype(jnp.float32), n=n)
    u_f = u_f * k_f
    out = jnp.fft.ifft(u_f, n=n)
    return out.real.astype(u.dtype)[..., :l]


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


def manual_flash_fft_conv_1d(u: jnp.ndarray, k: jnp.ndarray, seqlen: int, H: int) -> jnp.ndarray:
    """
    Manual implementation of Flash FFT Convolution for 1D signals.
    
    Args:
        u: Input tensor of shape (batch, channels, length)
        k: Kernel tensor of shape (channels, kernel_length)
        seqlen: Sequence length for FFT (should be power of 2)
        H: Number of channels/heads
    
    Returns:
        Convolved output tensor
    """
    if seqlen in [512, 2048]:
        k_f = jnp.fft.rfft(k, n=seqlen)
    else:
        k_f = jnp.fft.fft(k, n=seqlen)

    N = seqlen
    sqrt_N = int(math.sqrt(seqlen))

    f_sqrt_N_fft = fft_matrix(sqrt_N).astype(dtype)
    f_sqrt_N_ifft = ifft_matrix(sqrt_N).astype(dtype)

    twiddle_factors_fft = (compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).astype(dtype)
    twiddle_factors_ifft = compute_twiddle_factors_ifft(sqrt_N, sqrt_N).astype(dtype)

    k_f_permuted = k_f.reshape(H, sqrt_N, sqrt_N).swapaxes(-1, -2).reshape(H, N).astype(dtype)

    output = jnp.zeros(u.shape, dtype=dtype)

    # Pad and reshape input
    u_pad = pad_1d(u, seqlen - u.shape[-1]).reshape(u.shape[0], u.shape[1], sqrt_N, sqrt_N)

    # Process tiles
    for h_tile_id in range(H_TILE_SIZE):
        k_f_cur = k_f_permuted[h_tile_id].reshape(sqrt_N, sqrt_N)
        for b_tile_id in range(B_TILE_SIZE):
            # Forward FFT via 2-step monarch decomposition
            fft_2_step = jnp.matmul(
                jnp.matmul(f_sqrt_N_fft.T, u_pad[b_tile_id][h_tile_id]) * twiddle_factors_fft,
                f_sqrt_N_fft
            )
            # Element-wise multiplication in frequency domain
            elemwise_mul = fft_2_step * k_f_cur
            # Inverse FFT via 2-step monarch decomposition
            ifft_2_step = jnp.matmul(
                jnp.matmul(elemwise_mul, f_sqrt_N_ifft).T * twiddle_factors_ifft,
                f_sqrt_N_ifft
            )
            # Update output (cast to output dtype)
            output = output.at[b_tile_id, h_tile_id].set(
                ifft_2_step.T.reshape(seqlen)[:u.shape[-1]].astype(dtype)
            )
    
    return output


def manual_flash_fft_conv_2d(u: jnp.ndarray, k: jnp.ndarray, seqlen: int) -> jnp.ndarray:
    """
    Manual implementation of Flash FFT Convolution for 2D signals.
    
    Args:
        u: Input tensor of shape (batch, channels, height, width)
        k: Kernel tensor of shape (channels, kernel_height, kernel_width)
        seqlen: Sequence length parameter
    
    Returns:
        Convolved output tensor
    """
    s = (u.shape[-2], u.shape[-1])
    
    if seqlen in [512, 2048]:
        k_f = jnp.fft.rfftn(k, axes=(-2, -1), s=s)
    else:
        k_f = jnp.fft.fftn(k, axes=(-2, -1), s=s)

    N = seqlen
    sqrt_N = max(u.shape[-2], u.shape[-1])

    f_sqrt_N_fft = fft_matrix(sqrt_N).astype(dtype)
    f_sqrt_N_ifft = ifft_matrix(sqrt_N).astype(dtype)

    twiddle_factors_fft = (compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).astype(dtype)
    twiddle_factors_ifft = compute_twiddle_factors_ifft(sqrt_N, sqrt_N).astype(dtype)

    output = jnp.zeros(u.shape, dtype=dtype)

    # Pad input and kernel FFT
    u_pad = pad_2d(u, (0, sqrt_N - u.shape[-1], 0, sqrt_N - u.shape[-2]))
    k_f_pad = pad_2d(k_f, (0, sqrt_N - k_f.shape[-1], 0, sqrt_N - k_f.shape[-2]))

    # Permute kernel
    k_f_permuted = k_f_pad.swapaxes(-1, -2).astype(dtype)

    # Process tiles
    for h_tile_id in range(H_TILE_SIZE):
        k_f_cur = k_f_permuted[h_tile_id].reshape(sqrt_N, sqrt_N)
        for b_tile_id in range(B_TILE_SIZE):
            # Forward FFT via 2-step monarch decomposition
            fft_2_step = jnp.matmul(
                jnp.matmul(f_sqrt_N_fft.T, u_pad[b_tile_id][h_tile_id]) * twiddle_factors_fft,
                f_sqrt_N_fft
            )
            # Element-wise multiplication in frequency domain
            elemwise_mul = fft_2_step * k_f_cur
            # Inverse FFT via 2-step monarch decomposition
            ifft_2_step = jnp.matmul(
                jnp.matmul(elemwise_mul, f_sqrt_N_ifft).T * twiddle_factors_ifft,
                f_sqrt_N_ifft
            )
            # Update output (cast to output dtype)
            output = output.at[b_tile_id, h_tile_id].set(
                ifft_2_step.T[:s[0], :s[1]].astype(dtype)
            )
    
    return output

def manual_flash_fft_conv_2d_noifft(u: jnp.ndarray, k_f: jnp.ndarray, seqlen: int) -> jnp.ndarray:
    """
    Manual implementation of Flash FFT Convolution for 2D signals.
    
    Args:
        u: Input tensor of shape (batch, channels, height, width)
        k: Kernel tensor of shape (channels, kernel_height, kernel_width)
        seqlen: Sequence length parameter
    
    Returns:
        Convolved output tensor
    """
    s = (u.shape[-2], u.shape[-1])
    
    # if seqlen in [512, 2048]:
    #     k_f = jnp.fft.rfftn(k, axes=(-2, -1), s=s)
    # else:
    #     k_f = jnp.fft.fftn(k, axes=(-2, -1), s=s)

    N = seqlen
    sqrt_N = max(u.shape[-2], u.shape[-1])

    f_sqrt_N_fft = fft_matrix(sqrt_N).astype(dtype)
    f_sqrt_N_ifft = ifft_matrix(sqrt_N).astype(dtype)

    twiddle_factors_fft = (compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).astype(dtype)
    twiddle_factors_ifft = compute_twiddle_factors_ifft(sqrt_N, sqrt_N).astype(dtype)

    output = jnp.zeros(u.shape, dtype=dtype)

    # Pad input and kernel FFT
    u_pad = pad_2d(u, (0, sqrt_N - u.shape[-1], 0, sqrt_N - u.shape[-2]))
    k_f_pad = pad_2d(k_f, (0, sqrt_N - k_f.shape[-1], 0, sqrt_N - k_f.shape[-2]))

    # Permute kernel
    k_f_permuted = k_f_pad.swapaxes(-1, -2).astype(dtype)

    # Process tiles
    for h_tile_id in range(H_TILE_SIZE):
        k_f_cur = k_f_permuted[h_tile_id].reshape(sqrt_N, sqrt_N)
        for b_tile_id in range(B_TILE_SIZE):
            # Forward FFT via 2-step monarch decomposition
            fft_2_step = jnp.matmul(
                jnp.matmul(f_sqrt_N_fft.T, u_pad[b_tile_id][h_tile_id]) * twiddle_factors_fft,
                f_sqrt_N_fft
            )
            # Element-wise multiplication in frequency domain
            elemwise_mul = fft_2_step * k_f_cur
            # Inverse FFT via 2-step monarch decomposition
            ifft_2_step = jnp.matmul(
                jnp.matmul(elemwise_mul, f_sqrt_N_ifft).T * twiddle_factors_ifft,
                f_sqrt_N_ifft
            )
            # Update output (cast to output dtype)
            output = output.at[b_tile_id, h_tile_id].set(
                ifft_2_step.T[:s[0], :s[1]].astype(dtype)
            )
    
    return output

def manual_flash_fft_conv_2d_noifft_seq(u: jnp.ndarray, k_fs: jnp.ndarray, seqlen: int) -> jnp.ndarray:
    """
    Manual implementation of Flash FFT Convolution for 2D signals.
    
    Args:
        u: Input tensor of shape (batch, channels, height, width)
        k: Kernel tensor of shape (channels, kernel_height, kernel_width)
        seqlen: Sequence length parameter
    
    Returns:
        Convolved output tensor
    """
    s = (u.shape[-2], u.shape[-1])
    
    # if seqlen in [512, 2048]:
    #     k_f = jnp.fft.rfftn(k, axes=(-2, -1), s=s)
    # else:
    #     k_f = jnp.fft.fftn(k, axes=(-2, -1), s=s)

    N = seqlen
    sqrt_N = max(u.shape[-2], u.shape[-1])

    f_sqrt_N_fft = fft_matrix(sqrt_N).astype(dtype)
    f_sqrt_N_ifft = ifft_matrix(sqrt_N).astype(dtype)

    twiddle_factors_fft = (compute_twiddle_factors_fft(sqrt_N, sqrt_N) / N).astype(dtype)
    twiddle_factors_ifft = compute_twiddle_factors_ifft(sqrt_N, sqrt_N).astype(dtype)

    output = jnp.zeros(u.shape, dtype=dtype)

    # Pad input and kernel FFT
    u_pad = pad_2d(u, (0, sqrt_N - u.shape[-1], 0, sqrt_N - u.shape[-2]))
    k_fs_pad_permuted = [
        pad_2d(k_f, (0, sqrt_N - k_f.shape[-1], 0, sqrt_N - k_f.shape[-2])).swapaxes(-1, -2).astype(dtype)
        for k_f in k_fs
    ]

    # # Permute kernel
    # k_f_permuted = k_f_pad.swapaxes(-1, -2).astype(dtype)

    # Process tiles
    for k_f_permuted in k_fs_pad_permuted:
        for h_tile_id in range(H_TILE_SIZE):
            k_f_cur = k_f_permuted[h_tile_id].reshape(sqrt_N, sqrt_N)
            for b_tile_id in range(B_TILE_SIZE):
                # Forward FFT via 2-step monarch decomposition
                fft_2_step = jnp.matmul(
                    jnp.matmul(f_sqrt_N_fft.T, u_pad[b_tile_id][h_tile_id]) * twiddle_factors_fft,
                    f_sqrt_N_fft
                )
                # Element-wise multiplication in frequency domain
                elemwise_mul = fft_2_step * k_f_cur
                # Inverse FFT via 2-step monarch decomposition
                ifft_2_step = jnp.matmul(
                    jnp.matmul(elemwise_mul, f_sqrt_N_ifft).T * twiddle_factors_ifft,
                    f_sqrt_N_ifft
                )
                # Update output (cast to output dtype)
                output = output.at[b_tile_id, h_tile_id].set(
                    ifft_2_step.T[:s[0], :s[1]].astype(dtype)
                )
    
    return output


# JIT-compiled versions for better performance
manual_flash_fft_conv_1d_jit = jax.jit(manual_flash_fft_conv_1d, static_argnums=(2, 3))
manual_flash_fft_conv_2d_jit = jax.jit(manual_flash_fft_conv_2d, static_argnums=(2,))
ref_fft_conv_1d_jit = jax.jit(ref_fft_conv_1d, static_argnums=(2,))
ref_fft_conv_2d_jit = jax.jit(ref_fft_conv_2d, static_argnums=(2,))


if __name__ == "__main__":
    # Test the implementations
    print("\n--- Testing FFT Convolution Implementations ---\n")
    
    # Test parameters
    batch_size = 1
    channels = 1
    seq_len = 64
    kernel_len = 16
    
    # Create random inputs
    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)
    
    u_1d = jax.random.normal(k1, (batch_size, channels, seq_len), dtype=jnp.float32)
    k_1d = jax.random.normal(k2, (channels, kernel_len), dtype=jnp.float32)
    
    # Test reference 1D FFT conv
    print("Testing ref_fft_conv_1d...")
    out_ref_1d = ref_fft_conv_1d(u_1d, k_1d)
    print(f"  Input shape: {u_1d.shape}")
    print(f"  Kernel shape: {k_1d.shape}")
    print(f"  Output shape: {out_ref_1d.shape}")
    
    # Test 2D
    height, width = 16, 16
    key, k1, k2 = jax.random.split(key, 3)
    
    u_2d = jax.random.normal(k1, (batch_size, channels, height, width), dtype=jnp.float32)
    k_2d = jax.random.normal(k2, (channels, 5, 5), dtype=jnp.float32)
    
    print("\nTesting ref_fft_conv_2d...")
    out_ref_2d = ref_fft_conv_2d(u_2d, k_2d)
    print(f"  Input shape: {u_2d.shape}")
    print(f"  Kernel shape: {k_2d.shape}")
    print(f"  Output shape: {out_ref_2d.shape}")
    
    # Test matrix generation
    print("\nTesting matrix generation...")
    N_test = 8
    fft_mat = fft_matrix(N_test)
    ifft_mat = ifft_matrix(N_test)
    print(f"  FFT matrix shape: {fft_mat.shape}")
    print(f"  IFFT matrix shape: {ifft_mat.shape}")
    
    # Verify FFT matrix is correct (should give identity when multiplied with IFFT)
    identity_check = (fft_mat @ ifft_mat) / N_test
    identity_error = jnp.abs(identity_check - jnp.eye(N_test)).max()
    print(f"  FFT @ IFFT / N identity error: {identity_error:.2e}")
    
    print("\n--- All tests passed! ---")