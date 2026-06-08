import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

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


def div(
    input: Tensor,
    other: Union[Tensor, float, int],
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Tensor] = None
) -> Tensor:
    """
    Divides each element of input by the corresponding element of other,
    with optional rounding mode and output tensor support.
    
    Args:
        input: the dividend tensor
        other: the divisor (tensor or scalar)
        rounding_mode: optional rounding mode ('trunc', 'floor', or None)
        out: optional output tensor
    
    Returns:
        the result tensor
    """
    
    # Validate rounding_mode
    if rounding_mode is not None and rounding_mode not in ('trunc', 'floor'):
        raise ValueError(f"rounding_mode must be 'trunc', 'floor', or None, got {rounding_mode}")
    
    # Use torch.div which handles broadcasting, type promotion, and all input types
    # torch.div natively supports rounding_mode parameter
    y = torch.div(input, other, rounding_mode=rounding_mode)
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch

def test_div():
    results = {}

    # Test case 1: input and other are scalars
    input1 = torch.tensor(6.0, device='cuda')
    other1 = torch.tensor(3.0, device='cuda')
    results["test_case_1"] = div(input1, other1)

    # Test case 2: input and other are tensors of the same shape
    input2 = torch.tensor([6.0, 9.0], device='cuda')
    other2 = torch.tensor([3.0, 3.0], device='cuda')
    results["test_case_2"] = div(input2, other2)

    # Test case 3: input is a tensor and other is a scalar
    input3 = torch.tensor([6.0, 9.0], device='cuda')
    other3 = 3.0
    results["test_case_3"] = div(input3, other3)

    # Test case 4: input and other are tensors with broadcasting
    input4 = torch.tensor([[6.0, 9.0], [12.0, 15.0]], device='cuda')
    other4 = torch.tensor([3.0, 3.0], device='cuda')
    results["test_case_4"] = div(input4, other4)

    return results

test_results = test_div()
