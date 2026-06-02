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
def _bessel_j1_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    abs_x = tl.abs(x)
    y = x * 0.5
    y2 = y * y
    
    # Series expansion for small |x|
    # J1(x) = y * (1 - y^2/2 + y^4/12 - y^6/120 + y^8/1680)
    series = 1.0 - y2 * 0.5 + y2 * y2 * 0.08333333333333333 - y2 * y2 * y2 * 0.008333333333333333 + y2 * y2 * y2 * y2 * 0.0005952380952380952
    j1_series = y * series
    
    # Asymptotic for large |x|
    # J1(x) ≈ (sin(x) - cos(x)) / sqrt(pi * |x|)
    sqrt_denom = tl.sqrt(3.141592653589793 * abs_x)
    j1_asym = (tl.sin(x) - tl.cos(x)) / sqrt_denom
    
    out = tl.where(abs_x < 4.0, j1_series, j1_asym)
    tl.store(out_ptr + offs, out, mask=mask)

def bessel_j1(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point() and not input.is_complex():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _bessel_j1_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    r = torch.special.bessel_j1(input)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

def test_bessel_j1():
    results = {}

    # Test case 1: Basic test with a single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = bessel_j1(input1)

    # Test case 2: Test with a tensor of multiple values
    input2 = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    results["test_case_2"] = bessel_j1(input2)

    # Test case 3: Test with a tensor of negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = bessel_j1(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.linspace(-5.0, 5.0, steps=10, device='cuda')
    results["test_case_4"] = bessel_j1(input4)

    return results

test_results = test_bessel_j1()
