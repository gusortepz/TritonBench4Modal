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
def _scaled_add_kernel(y_ptr, x_ptr, out_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    z = y + alpha * x
    tl.store(out_ptr + offs, z, mask=mask)

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    if y.is_cuda and x.is_cuda and y.is_floating_point() and x.is_floating_point() and y.shape == x.shape:
        y_c = y.contiguous()
        x_c = x.contiguous()
        n = y_c.numel()
        out = torch.empty_like(y_c)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _scaled_add_kernel[grid](y_c, x_c, out, n, alpha, BLOCK_SIZE=1024)
        return torch.linalg.norm(out)
    return torch.linalg.norm(y + alpha * x)

##################################################################################################################################################



import torch

def test_scaled_add_norm():
    results = {}

    # Test case 1: Basic test with small tensors
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_norm(y1, x1, alpha1).item()

    # Test case 2: Test with negative alpha
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha2 = -1.0
    results["test_case_2"] = scaled_add_norm(y2, x2, alpha2).item()

    # Test case 3: Test with zero alpha
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x3 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha3 = 0.0
    results["test_case_3"] = scaled_add_norm(y3, x3, alpha3).item()

    # Test case 4: Test with zero vector x
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha4 = 2.0
    results["test_case_4"] = scaled_add_norm(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_norm()
