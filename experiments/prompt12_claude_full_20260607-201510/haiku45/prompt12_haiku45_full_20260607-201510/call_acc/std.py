import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union, Tuple

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


def std(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
    out: Optional[Tensor] = None
) -> Tensor:
    """
    Calculates the standard deviation over the specified dimensions of the input tensor.
    
    Args:
        input: the input tensor
        dim: the dimension or dimensions to reduce (int, tuple of ints, or None)
        correction: difference between sample size and degrees of freedom (default: 1 for Bessel's correction)
        keepdim: whether the output tensor has dim retained or not
        out: optional output tensor
        
    Returns:
        Standard deviation tensor
    """
    # Compute standard deviation using PyTorch reference
    # torch.std computes unbiased std with Bessel's correction by default
    y = torch.std(input, dim=dim, correction=correction, keepdim=keepdim)
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch

def test_std():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_1"] = std(input_tensor)

    # Test case 2: Test with dim parameter
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_2"] = std(input_tensor, dim=0)

    # Test case 3: Test with keepdim=True
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = std(input_tensor, dim=1, keepdim=True)

    # Test case 4: Test with correction=0 (population standard deviation)
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = std(input_tensor, correction=0)

    return results

test_results = test_std()
