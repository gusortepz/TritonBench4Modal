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
def _reciprocal_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    y = 1.0 / x
    tl.store(out_ptr + offs, y, mask=mask)

def reciprocal(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        if n == 0:
            return out if out is not None else x_c
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _reciprocal_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    r = torch.reciprocal(input)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

def test_reciprocal():
    results = {}

    # Test case 1: Basic test with positive and negative numbers
    a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298], device='cuda')
    results["test_case_1"] = reciprocal(a)

    # Test case 2: Test with a tensor containing zero (expecting inf)
    b = torch.tensor([0.0, 1.0, -1.0, 2.0], device='cuda')
    results["test_case_2"] = reciprocal(b)

    # Test case 3: Test with a tensor containing large numbers
    c = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_3"] = reciprocal(c)

    # Test case 4: Test with a tensor of ones (expecting ones)
    d = torch.ones(4, device='cuda')
    results["test_case_4"] = reciprocal(d)

    return results

test_results = test_reciprocal()
