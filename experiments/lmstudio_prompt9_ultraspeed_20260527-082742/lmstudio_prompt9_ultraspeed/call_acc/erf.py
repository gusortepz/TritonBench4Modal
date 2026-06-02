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
def _erf_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.erf(x)
    tl.store(out_ptr + offs, y, mask=mask)

def erf(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point() and not input.is_complex():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _erf_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    r = torch.erf(input, out=out)
    return r

##################################################################################################################################################



import torch

# def erf(input_tensor):
#     """
#     计算输入张量的误差函数（error function）。

#     参数：
#     input_tensor (Tensor): 输入的张量。

#     返回：
#     Tensor: 输入张量中每个元素的误差函数值。
#     """
#     return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)
    
    return results

test_results = test_erf()
