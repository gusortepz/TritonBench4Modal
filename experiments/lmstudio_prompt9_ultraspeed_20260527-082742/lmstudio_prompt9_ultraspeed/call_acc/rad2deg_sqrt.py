import torch
import triton
import triton.language as tl
from typing import Tuple

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@triton.jit
def _rad2deg_sqrt_kernel(x_ptr, deg_ptr, sqrt_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    deg = x * 57.29577951308232
    s = tl.sqrt(x)
    tl.store(deg_ptr + offs, deg, mask=mask)
    tl.store(sqrt_ptr + offs, s, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point():
        x_c = input.contiguous()
        n = x_c.numel()
        deg = torch.empty_like(x_c)
        sqrt = torch.empty_like(x_c)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _rad2deg_sqrt_kernel[grid](x_c, deg, sqrt, n, BLOCK_SIZE=1024)
        return deg, sqrt
    return input * 57.29577951308232, torch.sqrt(input)

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
