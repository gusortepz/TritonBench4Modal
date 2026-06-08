import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple, Union

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
    output_ptr,
    n: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: y += alpha * x, then compute dot(y, y).
    Accumulates the dot product across blocks.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    y_updated = y + alpha * x
    tl.store(y_ptr + offsets, y_updated, mask=mask)

    dot_product = tl.sum(y_updated * y_updated)
    tl.atomic_add(output_ptr, dot_product)


def scaled_add_dot(y: Tensor, x: Tensor, alpha: float) -> Tensor:
    """
    Computes y += alpha * x and returns the dot product of the modified y with itself.
    
    Args:
        y: Target tensor to be modified, shape (n,).
        x: Tensor to be scaled and added to y, shape (n,).
        alpha: Scalar multiplier for x.
    
    Returns:
        Dot product of the modified y with itself.
    """
    assert y.is_cuda, "Tensor must be on CUDA device"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert y.dtype in [torch.float32, torch.float64], "Tensor must be float32 or float64"
    assert x.dtype in [torch.float32, torch.float64], "Tensor must be float32 or float64"
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.dim() == 1, "y must be 1D"

    n = y.shape[0]
    dtype = y.dtype

    output = torch.zeros(1, dtype=dtype, device=y.device)

    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    _scaled_add_dot_kernel[(num_blocks,)](
        y,
        x,
        output,
        n=n,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.squeeze(0)

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
