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
def _digamma_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    mask_pos = x > 0.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    inv_x4 = inv_x2 * inv_x2
    
    res = tl.log(x) - 0.5 * inv_x - (1.0 / 12.0) * inv_x2 + (1.0 / 120.0) * inv_x4
    res = tl.where(mask_pos, res, -float('inf'))
    
    tl.store(out_ptr + offs, res, mask=mask)

def digamma(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point() and not input.is_complex():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _digamma_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    r = torch.digamma(input)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

# def digamma(input_tensor):
#     """
#     Computes the digamma function (logarithmic derivative of the gamma function) for the input tensor.

#     Args:
#     - input_tensor (torch.Tensor): The tensor on which to compute the digamma function.

#     Returns:
#     - torch.Tensor: A tensor containing the digamma values.
#     """
#     return torch.special.digamma(input_tensor)

def test_digamma():
    results = {}
    
    # Test case 1: Single positive value
    input_tensor = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = digamma(input_tensor)
    
    # Test case 2: Single negative value
    input_tensor = torch.tensor([-1.0], device='cuda')
    results["test_case_2"] = digamma(input_tensor)
    
    # Test case 3: Multiple positive values
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_3"] = digamma(input_tensor)
    
    # Test case 4: Mixed positive and negative values
    input_tensor = torch.tensor([1.0, -1.0, 2.0, -2.0], device='cuda')
    results["test_case_4"] = digamma(input_tensor)
    
    return results

test_results = test_digamma()
