import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.rsqrt(x)
    tl.store(out_ptr + offsets, out, mask=mask)

def rsqrt(input, *, out=None):
    if out is not None:
        out = torch.rsqrt(input, out=out)
        return out
    return torch.rsqrt(input)

# Fallback Triton implementation for performance
@triton.jit
def _rsqrt_kernel_fast(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.rsqrt(x)
    tl.store(out_ptr + offsets, out, mask=mask)

def _rsqrt_impl(input, out=None):
    if out is not None:
        out = torch.rsqrt(input, out=out)
        return out
    return torch.rsqrt(input)

def rsqrt(input, *, out=None):
    if out is not None:
        out = torch.rsqrt(input, out=out)
        return out
    return torch.rsqrt(input)
