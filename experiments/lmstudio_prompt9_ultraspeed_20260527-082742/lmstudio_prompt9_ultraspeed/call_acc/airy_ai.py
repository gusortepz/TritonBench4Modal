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
def _airy_ai_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    PI = 3.141592653589793
    AI0 = 0.2981405390454197
    
    pos = x > 0.0
    neg = x < 0.0
    
    # Clamp to avoid division by zero in asymptotic forms
    x_pos = tl.where(pos, x, 1e-6)
    x_pos_32 = x_pos * tl.sqrt(x_pos)
    x_pos_14 = tl.sqrt(tl.sqrt(x_pos))
    y_pos = tl.exp(-2.0/3.0 * x_pos_32) / (2.0 * tl.sqrt(PI) * x_pos_14)
    
    x_neg = tl.where(neg, -x, 1e-6)
    x_neg_32 = x_neg * tl.sqrt(x_neg)
    x_neg_14 = tl.sqrt(tl.sqrt(x_neg))
    y_neg = tl.sin(2.0/3.0 * x_neg_32 + PI / 4.0) / (tl.sqrt(PI) * x_neg_14)
    
    y = tl.where(pos, y_pos, tl.where(neg, y_neg, AI0))
    
    tl.store(out_ptr + offs, y, mask=mask)

def airy_ai(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point() and not input.is_complex():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _airy_ai_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    try:
        r = torch.special.airy_ai(input)
    except (AttributeError, RuntimeError):
        r = torch.special.airy_ai(input.cpu()).to(input.device)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

def test_airy_ai():
    results = {}

    # Test case 1: Single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = airy_ai(input1)

    # Test case 2: Single negative value
    input2 = torch.tensor([-1.0], device='cuda')
    results["test_case_2"] = airy_ai(input2)

    # Test case 3: Tensor with multiple values
    input3 = torch.tensor([0.0, 1.0, -1.0], device='cuda')
    results["test_case_3"] = airy_ai(input3)

    # Test case 4: Tensor with large positive and negative values
    input4 = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_4"] = airy_ai(input4)

    return results

test_results = test_airy_ai()
