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

_RAD2DEG = 180.0 / 3.141592653589793

@triton.jit
def _rad2deg_sqrt_kernel(
    in_ptr,
    deg_ptr,
    sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    deg = x * 57.29577951308232  # 180.0 / pi
    sq = tl.sqrt(x)
    tl.store(deg_ptr + offsets, deg, mask=mask)
    tl.store(sqrt_ptr + offsets, sq, mask=mask)


def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not input.is_cuda or not input.is_floating_point():
        # PyTorch fallback
        deg = torch.rad2deg(input)
        sq = torch.sqrt(input)
        return deg, sq

    input_flat = input.contiguous().view(-1)
    n = input_flat.numel()
    deg_out = torch.empty_like(input_flat)
    sqrt_out = torch.empty_like(input_flat)

    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    try:
        _rad2deg_sqrt_kernel[grid](
            input_flat,
            deg_out,
            sqrt_out,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return deg_out.view(input.shape), sqrt_out.view(input.shape)
    except Exception:
        deg = torch.rad2deg(input)
        sq = torch.sqrt(input)
        return deg, sq

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
