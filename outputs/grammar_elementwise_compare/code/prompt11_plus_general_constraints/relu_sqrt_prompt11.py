import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from torch import Tensor

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
def _relu_sqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    out = tl.sqrt(x)
    tl.store(out_ptr + offsets, out, mask=mask)

def relu_sqrt(input: Tensor, inplace: bool = False, out: Optional[Tensor] = None) -> Tensor:
    n_elements = input.numel()
    if n_elements == 0:
        if out is not None:
            return out
        return input.clone()
    
    if inplace:
        if out is not None:
            out.copy_(input)
            input = out
        else:
            input = input.clone()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _relu_sqrt_kernel[grid](input, input, n_elements, BLOCK_SIZE=1024)
        return input
    else:
        if out is not None:
            result = out
        else:
            result = torch.empty_like(input)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _relu_sqrt_kernel[grid](input, result, n_elements, BLOCK_SIZE=1024)
        return result
