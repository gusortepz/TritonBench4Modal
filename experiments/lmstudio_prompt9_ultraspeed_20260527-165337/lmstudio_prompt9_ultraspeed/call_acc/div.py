import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=1.0)
    z = x / y
    tl.store(out_ptr + offs, z, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and input.is_cuda and other.is_cuda and input.is_floating_point() and other.is_floating_point() and (not input.is_complex()) and (not other.is_complex()) and input.shape == other.shape:
        x_c = input.contiguous()
        y_c = other.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == input.shape and out.dtype == input.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _div_kernel[grid](x_c, y_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    else:
        if rounding_mode is not None:
            if rounding_mode == "floor":
                return torch.floor_divide(input, other, out=out)
            elif rounding_mode == "trunc":
                return torch.trunc_divide(input, other, out=out)
            else:
                raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")
        return torch.div(input, other, out=out)

##################################################################################################################################################



import torch

def test_div():
    results = {}

    # Test case 1: input and other are scalars
    input1 = torch.tensor(6.0, device='cuda')
    other1 = torch.tensor(3.0, device='cuda')
    results["test_case_1"] = div(input1, other1)

    # Test case 2: input and other are tensors of the same shape
    input2 = torch.tensor([6.0, 9.0], device='cuda')
    other2 = torch.tensor([3.0, 3.0], device='cuda')
    results["test_case_2"] = div(input2, other2)

    # Test case 3: input is a tensor and other is a scalar
    input3 = torch.tensor([6.0, 9.0], device='cuda')
    other3 = 3.0
    results["test_case_3"] = div(input3, other3)

    # Test case 4: input and other are tensors with broadcasting
    input4 = torch.tensor([[6.0, 9.0], [12.0, 15.0]], device='cuda')
    other4 = torch.tensor([3.0, 3.0], device='cuda')
    results["test_case_4"] = div(input4, other4)

    return results

test_results = test_div()
