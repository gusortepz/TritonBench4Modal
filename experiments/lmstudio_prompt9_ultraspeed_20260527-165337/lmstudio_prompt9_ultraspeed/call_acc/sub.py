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
def _sub_kernel(input_ptr, other_ptr, out_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    input_val = tl.load(input_ptr + offs, mask=mask, other=0.0)
    other_val = tl.load(other_ptr + offs, mask=mask, other=0.0)
    out_val = input_val - alpha * other_val
    tl.store(out_ptr + offs, out_val, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and input.is_cuda and other.is_cuda and input.is_floating_point() and other.is_floating_point() and (not input.is_complex()) and (not other.is_complex()) and input.shape == other.shape:
        input_c = input.contiguous()
        other_c = other.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == input.shape and out.dtype == input.dtype and out.is_contiguous() else torch.empty_like(input_c)
        n = input_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _sub_kernel[grid](input_c, other_c, out_t, n, alpha, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    else:
        y = torch.sub(input, other, alpha=alpha)
        if out is not None:
            out.copy_(y)
            return out
        return y

##################################################################################################################################################



import torch

def test_sub():
    results = {}

    # Test case 1: Basic subtraction with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub(input1, other1)

    # Test case 2: Subtraction with alpha
    input2 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    other2 = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    results["test_case_2"] = sub(input2, other2, alpha=2)

    # Test case 3: Subtraction with a scalar other
    input3 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    other3 = 2.0
    results["test_case_3"] = sub(input3, other3)

    # Test case 4: Subtraction with out parameter
    input4 = torch.tensor([10.0, 11.0, 12.0], device='cuda')
    other4 = torch.tensor([3.0, 3.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = sub(input4, other4, out=out4)

    return results

test_results = test_sub()
