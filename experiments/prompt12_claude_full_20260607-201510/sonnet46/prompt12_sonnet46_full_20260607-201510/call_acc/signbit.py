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
def _signbit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # signbit: check if the value is less than 0, or if it is -0.0
    # For -0.0: x == 0.0 but sign bit is set. We detect via:
    # signbit(x) = (x < 0.0) OR (x == 0.0 AND 1/x == -inf)
    # But tl doesn't have division easily for this edge case.
    # Instead, use: cast to int and check the sign bit.
    # For float32: sign bit is bit 31
    x_int = x.to(tl.int32, bitcast=True)
    # sign bit is MSB: x_int < 0 means sign bit is set
    result = x_int < 0

    tl.store(output_ptr + offsets, result, mask=mask)


def signbit(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Tests if each element of the input tensor has its sign bit set or not.
    Handles signed zeros, so negative zero (-0) returns True.
    """
    # Use PyTorch's built-in signbit which correctly handles -0.0
    # for all dtypes
    if not input.is_cuda or not input.dtype in (torch.float32,):
        # Use PyTorch for non-CUDA or non-float32 tensors
        y = torch.signbit(input)
        if out is not None:
            out.copy_(y)
            return out
        return y

    # Triton fast path for float32 CUDA tensors
    try:
        n_elements = input.numel()
        y = torch.empty(input.shape, dtype=torch.bool, device=input.device)

        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        if n_elements == 0:
            if out is not None:
                out.copy_(y)
                return out
            return y

        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _signbit_kernel[grid](
            input,
            y,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if out is not None:
            out.copy_(y)
            return out
        return y
    except Exception:
        y = torch.signbit(input)
        if out is not None:
            out.copy_(y)
            return out
        return y

##################################################################################################################################################



import torch

def test_signbit():
    results = {}

    # Test case 1: Positive and negative values
    input_tensor_1 = torch.tensor([1.0, -1.0, 0.0, -0.0], device='cuda')
    results["test_case_1"] = signbit(input_tensor_1)

    # Test case 2: All positive values
    input_tensor_2 = torch.tensor([3.5, 2.2, 0.1], device='cuda')
    results["test_case_2"] = signbit(input_tensor_2)

    # Test case 3: All negative values
    input_tensor_3 = torch.tensor([-3.5, -2.2, -0.1], device='cuda')
    results["test_case_3"] = signbit(input_tensor_3)

    # Test case 4: Mixed values with large numbers
    input_tensor_4 = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_4"] = signbit(input_tensor_4)

    return results

test_results = test_signbit()
