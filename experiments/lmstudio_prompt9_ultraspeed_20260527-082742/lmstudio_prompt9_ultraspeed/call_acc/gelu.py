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

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n, approximate: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate:
        tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        tanh_val = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
        out = 0.5 * x * (1.0 + tanh_val)
    else:
        out = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
        
    tl.store(out_ptr + offsets, out, mask=mask)

def gelu(input, approximate='none', *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point():
        x = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous() else torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _gelu_kernel[grid](x, out_t, n, approximate=(approximate == 'tanh'), BLOCK_SIZE=BLOCK_SIZE)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    return F.gelu(input, approximate=approximate, out=out)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def gelu(input: torch.Tensor, approximate: str='none') -> torch.Tensor:
#     return F.gelu(input, approximate=approximate)

def test_gelu():
    results = {}
    
    # Test case 1: Default approximate='none'
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = gelu(input_tensor_1)
    
    # Test case 2: approximate='tanh'
    input_tensor_2 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_2"] = gelu(input_tensor_2, approximate='tanh')
    
    # Test case 3: Larger tensor with default approximate='none'
    input_tensor_3 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_3"] = gelu(input_tensor_3)
    
    # Test case 4: Larger tensor with approximate='tanh'
    input_tensor_4 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    results["test_case_4"] = gelu(input_tensor_4, approximate='tanh')
    
    return results

test_results = test_gelu()
