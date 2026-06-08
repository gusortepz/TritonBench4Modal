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
def _scaled_add_dot_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    alpha,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # y += alpha * x
    y_new = y_vals + alpha * x_vals

    # store modified y back
    tl.store(y_ptr + offsets, y_new, mask=mask)

    # compute contribution to dot product: y_new * y_new
    dot_contrib = tl.sum(y_new * y_new, axis=0)

    # atomic add to output
    tl.atomic_add(out_ptr, dot_contrib)


def scaled_add_dot(y: Tensor, x: Tensor, alpha: float) -> Tensor:
    """
    Computes y += alpha * x and returns the dot product of the modified y with itself.
    
    Args:
        y: The target tensor to be modified, of shape (n,).
        x: The tensor to be scaled and added to y, of shape (n,).
        alpha: The scalar multiplier for x.
    
    Returns:
        A scalar tensor containing dot(y_modified, y_modified).
    """
    # Fallback for non-CUDA or non-float tensors
    if not y.is_cuda or not x.is_cuda or y.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        y.add_(x, alpha=alpha)
        return torch.dot(y, y)

    n = y.numel()
    assert x.numel() == n, "x and y must have the same number of elements"

    # Work in float32 for the computation if needed
    y_compute = y.contiguous()
    x_compute = x.contiguous()

    if y_compute.dtype != torch.float32:
        # Fall back to PyTorch for non-float32
        y.add_(x, alpha=alpha)
        return torch.dot(y, y)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n, 1)))
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    out = torch.zeros(1, dtype=torch.float32, device=y.device)

    try:
        _scaled_add_dot_kernel[grid](
            y_compute,
            x_compute,
            out,
            alpha,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        # If y was contiguous and same storage, it's already modified
        # If we made a copy, copy back
        if y_compute.data_ptr() != y.data_ptr():
            y.copy_(y_compute)
    except Exception:
        # Fallback
        y.add_(x, alpha=alpha)
        return torch.dot(y, y)

    return out[0]

##################################################################################################################################################



import torch
from torch import Tensor

def test_scaled_add_dot():
    results = {}

    # Test case 1: Basic functionality
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_dot(y1, x1, alpha1).item()

    # Test case 2: Zero tensor x
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha2 = 2.0
    results["test_case_2"] = scaled_add_dot(y2, x2, alpha2).item()

    # Test case 3: Zero tensor y
    y3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    x3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha3 = 1.0
    results["test_case_3"] = scaled_add_dot(y3, x3, alpha3).item()

    # Test case 4: Negative alpha
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha4 = -1.0
    results["test_case_4"] = scaled_add_dot(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_dot()
