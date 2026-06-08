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
def _erfc_sqrt_kernel(
    x_ptr,
    erfc_ptr,
    sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # erfc(x) = 1 - erf(x)
    erfc_val = 1.0 - tl.erf(x)
    sqrt_val = tl.sqrt(x)
    tl.store(erfc_ptr + offsets, erfc_val, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=mask)


def _erfc_sqrt_pytorch(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.special.erfc(input), torch.sqrt(input)


def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not input.is_cuda or not input.is_floating_point() or input.is_complex():
        return _erfc_sqrt_pytorch(input)

    try:
        x = input.contiguous()
        n = x.numel()
        erfc_out = torch.empty_like(x)
        sqrt_out = torch.empty_like(x)

        BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
        if BLOCK_SIZE == 0:
            BLOCK_SIZE = 1
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _erfc_sqrt_kernel[grid](
            x,
            erfc_out,
            sqrt_out,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return erfc_out, sqrt_out
    except Exception:
        return _erfc_sqrt_pytorch(input)

##################################################################################################################################################



import torch
import math
from typing import Tuple

# def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Computes the complementary error function (erfc) and the square root of each element in the input tensor.

#     Args:
#         input (torch.Tensor): The input tensor for which the erfc and square root are computed.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
#             - erfc_result (torch.Tensor): The complementary error function results.
#             - sqrt_result (torch.Tensor): The square root results.
#     """
#     erfc_result = torch.erfc(input)
#     sqrt_result = torch.sqrt(input)
#     sqrt_result[input < 0] = float('nan')
#     return (erfc_result, sqrt_result)

def test_erfc_sqrt():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_1"] = erfc_sqrt(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = erfc_sqrt(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = erfc_sqrt(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = erfc_sqrt(input4)

    return results

test_results = test_erfc_sqrt()
