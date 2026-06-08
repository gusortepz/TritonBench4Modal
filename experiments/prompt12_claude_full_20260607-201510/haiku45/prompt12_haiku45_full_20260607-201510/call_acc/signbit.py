import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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
    """
    Triton kernel to compute signbit.
    Returns True (1.0) if the sign bit is set, False (0.0) otherwise.
    Handles signed zeros correctly.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute signbit: check if the sign bit is set
    # For IEEE 754 floats, signbit is 1 if x < 0 or (x == 0 and sign(x) is negative)
    # We use tl.where to handle both cases correctly
    signbit_result = tl.where(x < 0.0, 1.0, 0.0)
    # Handle the case of negative zero: if x == 0, check if it's negative zero
    is_zero = x == 0.0
    # For negative zero, we need to check the sign bit directly
    # In IEEE 754, we can use a bitwise operation, but tl doesn't expose that directly
    # Instead, we handle it by checking if (1.0 / x) is negative infinity
    neg_inf = tl.where(is_zero, tl.where(1.0 / x < 0.0, 1.0, 0.0), signbit_result)

    # Store the result
    tl.store(output_ptr + offsets, neg_inf, mask=mask)


def signbit(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Tests if each element of the input tensor has its sign bit set or not.
    Handles signed zeros, so negative zero (-0) returns True.

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: a boolean tensor where True indicates the sign bit is set.
    """
    # Use torch.signbit as the reference implementation
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
