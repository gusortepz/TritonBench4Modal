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


def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None) -> Tensor:
    """
    Performs a fused operation combining horizontal stacking (hstack) and
    element-wise division. First horizontally stacks the sequence of tensors,
    then divides each element of the resulting tensor by the corresponding
    element of the divisor tensor, with optional rounding modes.
    
    Args:
        tensors: Sequence of tensors to be horizontally stacked.
        divisor: Tensor or Number to divide by. Must be broadcastable.
        rounding_mode: None, 'trunc', or 'floor'.
        out: Optional output tensor.
    
    Returns:
        Tensor: Result of hstack followed by division.
    """
    # Step 1: Horizontally stack the tensors
    stacked = torch.hstack(tensors)
    
    # Step 2: Divide with optional rounding mode
    y = torch.div(stacked, divisor, rounding_mode=rounding_mode)
    
    # Step 3: Handle optional out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch

def test_fused_hstack_div():
    results = {}

    # Test case 1: Basic functionality with two tensors and a scalar divisor
    tensors1 = [torch.tensor([1, 2], device='cuda'), torch.tensor([3, 4], device='cuda')]
    divisor1 = 2
    results["test_case_1"] = fused_hstack_div(tensors1, divisor1)

    # Test case 3: Using rounding_mode='floor'
    tensors3 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor3 = 2
    results["test_case_3"] = fused_hstack_div(tensors3, divisor3, rounding_mode='floor')

    # Test case 4: Using rounding_mode='trunc'
    tensors4 = [torch.tensor([1.5, 2.5], device='cuda'), torch.tensor([3.5, 4.5], device='cuda')]
    divisor4 = 2
    results["test_case_4"] = fused_hstack_div(tensors4, divisor4, rounding_mode='trunc')

    return results

test_results = test_fused_hstack_div()
