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
def _i0_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    abs_x = tl.abs(x)
    
    t = (x * 0.5) * (x * 0.5)
    p_small = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    
    inv_x = 1.0 / abs_x
    p_large = 0.91628072 + inv_x * (-0.02057706 + inv_x * (0.002635537 + inv_x * (-0.000164024 + inv_x * 0.0000481353)))
    p_large = tl.exp(abs_x) * tl.rsqrt(6.283185307179586 * abs_x) * p_large
    
    y = tl.where(abs_x <= 3.75, p_small, p_large)
    tl.store(out_ptr + offs, y, mask=mask)

def i0(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point() and not input.is_complex():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _i0_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    r = torch.special.i0(input)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

def test_i0():
    results = {}

    # Test case 1: Simple tensor on GPU
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = i0(input_tensor_1)

    # Test case 2: Larger tensor with negative values on GPU
    input_tensor_2 = torch.tensor([-1.0, -2.0, 3.0, 4.0], device='cuda')
    results["test_case_2"] = i0(input_tensor_2)

    # Test case 3: Tensor with mixed positive and negative values on GPU
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = i0(input_tensor_3)

    # Test case 4: Tensor with fractional values on GPU
    input_tensor_4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = i0(input_tensor_4)

    return results

test_results = test_i0()
