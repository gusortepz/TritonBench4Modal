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
def _logit_kernel(x_ptr, out_ptr, n, eps, clamp, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    if clamp:
        x = tl.maximum(tl.minimum(x, 1.0 - eps), eps)
        
    out = tl.log(x) - tl.log(1.0 - x)
    tl.store(out_ptr + offs, out, mask=mask)

def logit(input, eps=None, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _logit_kernel[grid](x_c, out_t, n, eps if eps is not None else 0.0, 1 if eps is not None else 0, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    if eps is not None:
        r = torch.logit(input, eps=eps)
    else:
        r = torch.logit(input)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

def test_logit():
    results = {}

    # Test case 1: Basic test with input tensor in range [0, 1] without eps
    input1 = torch.tensor([0.2, 0.5, 0.8], device='cuda')
    results["test_case_1"] = logit(input1)

    # Test case 2: Test with input tensor in range [0, 1] with eps
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    eps = 1e-6
    results["test_case_2"] = logit(input2, eps=eps)

    # Test case 3: Test with input tensor in range [0, 1] with eps and out tensor
    input3 = torch.tensor([0.1, 0.9], device='cuda')
    out = torch.empty_like(input3)
    results["test_case_3"] = logit(input3, eps=eps, out=out)

    # Test case 4: Test with input tensor in range [0, 1] with out tensor
    input4 = torch.tensor([0.3, 0.7], device='cuda')
    out = torch.empty_like(input4)
    results["test_case_4"] = logit(input4, out=out)

    return results

test_results = test_logit()
