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


def add_mean(
    input: Tensor,
    other: Union[Tensor, float, int],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    alpha: Union[float, int] = 1,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Adds the `other` tensor (scaled by `alpha`) to the `input` tensor,
    then computes the mean along the specified dimension(s).
    
    Supports broadcasting, type promotion, and works with integer, float, and complex inputs.
    """
    # Cast input if dtype is specified
    if dtype is not None:
        x = input.to(dtype=dtype)
    else:
        x = input
    
    # Handle other as either Tensor or scalar
    if isinstance(other, Tensor):
        other_scaled = other * alpha
    else:
        other_scaled = other * alpha
    
    # Add other to input
    result = x + other_scaled
    
    # Compute mean along dimension(s)
    y = torch.mean(result, dim=dim, keepdim=keepdim)
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch

def test_add_mean():
    results = {}

    # Test case 1: Basic addition and mean with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_mean(input1, other1)

    # Test case 2: Addition with scalar other and non-default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = add_mean(input2, other2, alpha=2)

    # Test case 3: Addition with mean along a specific dimension
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = add_mean(input3, other3, dim=0)

    # Test case 4: Addition with mean and keepdim=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = add_mean(input4, other4, dim=1, keepdim=True)

    return results

test_results = test_add_mean()
