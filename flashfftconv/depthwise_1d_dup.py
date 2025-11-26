# Copyright (c) 2023, Dan Fu and Hermann Kumbong.
import torch
import math
# from monarch_cuda import conv1d_forward, conv1d_backward
from einops import rearrange

BX = 256
BY = 1
BZ = 1

TILE_SIZE_L = 4
TILE_SIZE_D = 1

blockDim = {
    'x': BX,
    'y': BY,
    'z': BZ,
}
threadIdx = {
    'x': 0,
    'y': 0, 
    'z': 0
}
blockIdx = {
    'x': 0,
    'y': 0,
    'z': 0,
}
gridDim = {}

def __hfma(a, b, c):
    return a * b + c

def _conv1d_k_3(u, weights, bias, padding, l, d, L, D, K):
    tmp = bias[d]

    idx = l - padding

    if idx >= 0 and idx < L:
        weight = weights[0]
        tmp = __hfma(u[d * L + idx], weight, tmp)
    
    idx += 1
    if idx >= 0 and idx < L:
        weight = weights[1]
        tmp = __hfma(u[d * L + idx], weight, tmp)

    idx += 1
    if idx >= 0 and idx < L:
        weight = weights[2]
        tmp = __hfma(u[d * L + idx], weight, tmp)

    return tmp

def conv1d_kernel(
    u, weights, bias, out, padding,
    B, L, D, K, L_out
):
    b = blockIdx.z * blockDim.z + threadIdx.z
    d = blockIdx.y * blockDim.y * TILE_SIZE_D + threadIdx.y
    l_offset = blockIdx.x * blockDim.x * TILE_SIZE_L + threadIdx.x
    
    # T tmp; 
    # T weight;

    # int idx;
    # int l;

    for l_tile in range(TILE_SIZE_L):
        l = l_offset + l_tile * blockDim.x

        tmp = bias[d]

        if d < D and l < L_out and b < B:
            if K == 3:
                out[b * L_out * D + d * L_out + l] = _conv1d_k_3(u + b * L * D, weights + d * K, bias, padding, l, d, L, D, K)
            else:
                for k in range(K):
                    idx = l - padding + k
                    if idx >= 0 and idx < L:
                        weight = weights[d * K + k]
                        tmp = __hfma(u[b * L_out * D + d * L + idx], weight, tmp)
                out[b * L_out * D + d * L_out + l] = tmp

def conv1d_cuda_bhl(u, weight, bias, padding):
    b = u.size(0)
    d = u.size(1)
    l = u.size(2)

    k = weight.size(1)

    l_out = (l + 2 * padding - k + 1)
    
    # dim3 blockDims(BX, BY, BZ);

    # dim3 gridDims(ceil(l_out * 1.0 / (BX * TILE_SIZE_L) ), ceil((d * 1.0) / (BY * TILE_SIZE_D)), ceil((b * 1.0) / BZ));
    threadIdx.x = math.ceil(l_out * 1.0 / (BX * TILE_SIZE_L) )
    threadIdx.y = math.ceil((d * 1.0) / (BY * TILE_SIZE_D))
    threadIdx.z = math.ceil(l_out * 1.0 / (BX * TILE_SIZE_L) )

    out = torch.zeros(b, d, l_out, dtype=dtype[0], device=device)

    # DISPATCH_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), weight.scalar_type(),
    #     "depthwise conv 1d fwd bhl",
    #     ([&]
    #         { conv1d_kernel<input_t, weight_t><<<gridDims, blockDims>>>(
    #                 static_cast<input_t *>(u.data_ptr()),
    #                 static_cast<weight_t *>(weight.data_ptr()),
    #                 static_cast<weight_t *>(bias.data_ptr()),
    #                 static_cast<input_t *>(out.data_ptr()),
    #                 padding,
    #                 b,
    #                 l,
    #                 d,
    #                 k,
    #                 l_out
    #                 ); 
    #         }
    #     )
    # );
    conv1d_kernel(u, weight, bias, out, padding, b, l, d, k, l_out)

    return out

class conv1dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, padding, is_bhl=True):
        # outputs = conv1d_forward(input, weights, bias, padding, is_bhl)
        outputs = conv1d_cuda_bhl(input, weights, bias, padding)
        ctx.padding = padding
        ctx.is_bhl = is_bhl
        ctx.save_for_backward(input, weights, bias)
        return outputs

    # @staticmethod
    # def backward(ctx, dout):
    #     input, weight, bias = ctx.saved_tensors
    #     dout  = dout.contiguous()
    #     du, dk, dbias = conv1d_backward(dout, input, weight, bias, ctx.padding, ctx.is_bhl)
    #     return du, dk, dbias, None, None
    
#TODO: initialization    
class FlashDepthWiseConv1dPython(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding, weights, bias, is_bhl=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FlashDepthWiseConv1dPython, self).__init__()
        self.d = channels
        self.k = kernel_size
        self.padding = padding
        self.is_bhl = is_bhl
        if is_bhl:
            self.weights  = torch.nn.Parameter(weights.squeeze())
        else:
            self.weights  = torch.nn.Parameter(rearrange(weights.squeeze(), 'd k -> k d').detach().clone().contiguous())
        self.bias = torch.nn.Parameter(bias.detach().clone().contiguous())
        self.reset_parameters(weights, bias)

    #TODO: initialization
    def reset_parameters(self, weights, bias):
        pass
        # stdv = 1.0 / math.sqrt(self.state_size)
        # for weight in self.parameters():
        #     weight.data.uniform_(-stdv, +stdv)
    
    #current format for the weights is transpose of what is used in nn.Conv1d
    #[HK]: load the weights for the conv1d layer and then transpose them
    def load_state_dict(self, state_dict, strict: bool = True):
        pass
    
    #[HK]: transpose the weights before saving so that they can be loaded in nn.Conv1d
    def save_state_dict(self):
        pass
    
    def forward(self, input):
        return conv1dFunc.apply(input, self.weights, self.bias, self.padding, self.is_bhl)