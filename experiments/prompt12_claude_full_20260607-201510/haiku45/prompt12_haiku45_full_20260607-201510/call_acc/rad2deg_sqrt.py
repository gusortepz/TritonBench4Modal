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
def _rad2deg_sqrt_kernel(
    input_ptr,
    deg_ptr,
    sqrt_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: convert radians to degrees and compute square root.
    rad2deg: x * (180.0 / pi)
    sqrt: tl.sqrt(x)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Convert radians to degrees: x * (180 / pi)
    deg_result = x * (180.0 / 3.141592653589793)

    # Compute square root
    sqrt_result = tl.sqrt(x)

    tl.store(deg_ptr + offsets, deg_result, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_result, mask=mask)


def rad2deg_sqrt(input: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute radians-to-degrees conversion and square root for each element.

    Args:
        input (Tensor): The input tensor with angles in radians.

    Returns:
        Tuple[Tensor, Tensor]: A tuple where the first element is the tensor
            with angles converted from radians to degrees, and the second
            element is the tensor with square roots of input elements.
    """
    # Validate input
    if not isinstance(input, Tensor):
        raise TypeError(f"input must be a Tensor, got {type(input)}")

    if not input.is_cuda or not input.dtype in (torch.float32, torch.float64):
        # Fall back to PyTorch for non-CUDA or non-float tensors
        deg = torch.rad2deg(input)
        sqrt_result = torch.sqrt(input)
        return deg, sqrt_result

    # Create output tensors
    deg_out = torch.empty_like(input)
    sqrt_out = torch.empty_like(input)

    numel = input.numel()
    if numel == 0:
        return deg_out, sqrt_out

    # Determine block size
    BLOCK_SIZE = min(triton.next_power_of_2(numel), 1024)

    # Calculate grid
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    # Launch kernel
    _rad2deg_sqrt_kernel[grid](
        input,
        deg_out,
        sqrt_out,
        numel=numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return deg_out, sqrt_out

##################################################################################################################################################



import torch
from typing import Tuple

# def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     deg_result = torch.rad2deg(input)
#     sqrt_result = torch.sqrt(input)
#     return (deg_result, sqrt_result)

def test_rad2deg_sqrt():
    results = {}

    # Test case 1: Basic test with positive radians
    a = torch.tensor([3.142, 1.570, 0.785, 0.0], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(a)
    results["test_case_1"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 2: Test with zero
    b = torch.tensor([0.0], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(b)
    results["test_case_2"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 3: Test with negative radians
    c = torch.tensor([-3.142, -1.570, -0.785], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(c)
    results["test_case_3"] = (deg_result.cpu(), sqrt_result.cpu())

    # Test case 4: Test with a mix of positive and negative radians
    d = torch.tensor([3.142, -1.570, 0.785, -0.785], device='cuda')
    deg_result, sqrt_result = rad2deg_sqrt(d)
    results["test_case_4"] = (deg_result.cpu(), sqrt_result.cpu())

    return results

test_results = test_rad2deg_sqrt()
