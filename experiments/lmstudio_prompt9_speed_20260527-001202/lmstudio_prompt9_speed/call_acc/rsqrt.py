import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.rsqrt(x)
    tl.store(out_ptr + offs, y, mask=mask)

def rsqrt(input, *, out=None):
    def ref():
        r = torch.rsqrt(input)
        if out is not None:
            out.copy_(r)
            return out
        return r

    if not isinstance(input, torch.Tensor):
        return ref()
    if not input.is_cuda or input.is_complex():
        return ref()
    if not input.is_floating_point():
        return ref()
    if out is not None and (not isinstance(out, torch.Tensor) or not out.is_cuda or out.shape != input.shape or out.dtype != input.dtype):
        return ref()

    input = input.contiguous()
    out_t = out if out is not None and out.is_contiguous() else torch.empty_like(input)
    n = input.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _rsqrt_kernel[grid](input, out_t, n, BLOCK_SIZE=1024)
    if out is not None and out_t is not out:
        out.copy_(out_t)
        return out
    return out_t

##################################################################################################################################################



import torch

def test_rsqrt():
    results = {}

    # Test case 1: Positive elements
    input1 = torch.tensor([4.0, 16.0, 25.0], device='cuda')
    results["test_case_1"] = rsqrt(input1)

    # Test case 2: Contains zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = rsqrt(input2)

    # Test case 3: Contains negative elements
    input3 = torch.tensor([-1.0, 4.0, 9.0], device='cuda')
    results["test_case_3"] = rsqrt(input3)

    # Test case 4: All elements are zero
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = rsqrt(input4)

    return results

test_results = test_rsqrt()
