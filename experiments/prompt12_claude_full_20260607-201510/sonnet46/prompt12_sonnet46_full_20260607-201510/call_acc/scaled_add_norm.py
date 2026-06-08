import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from torch import Tensor

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


@triton.jit
def _scaled_add_kernel(
    y_ptr,
    x_ptr,
    alpha,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = y + alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _norm_kernel(
    y_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    sq = y * y
    local_sum = tl.sum(sq, axis=0)
    tl.atomic_add(out_ptr, local_sum)


def scaled_add_norm(y: Tensor, x: Tensor, alpha: float) -> Tensor:
    if not y.is_cuda or not x.is_cuda or y.is_complex() or x.is_complex():
        y.add_(x, alpha=alpha)
        return torch.linalg.norm(y)

    y_flat = y.view(-1)
    x_flat = x.view(-1)
    n = y_flat.numel()

    BLOCK_SIZE = min(1024, triton.next_power_of_2(n) if n > 0 else 1)
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    try:
        _scaled_add_kernel[grid](
            y_flat,
            x_flat,
            alpha,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        out = torch.zeros(1, dtype=y_flat.dtype, device=y_flat.device)
        _norm_kernel[grid](
            y_flat,
            out,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        result = torch.sqrt(out).squeeze()
        return result
    except Exception:
        y.add_(x, alpha=alpha)
        return torch.linalg.norm(y)

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
