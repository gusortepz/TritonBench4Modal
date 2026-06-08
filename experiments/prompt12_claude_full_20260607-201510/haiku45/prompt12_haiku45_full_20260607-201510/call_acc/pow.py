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


def pow(input: Tensor, exponent: Union[float, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Takes the power of each element in input with exponent and returns a tensor with the result.
    
    Args:
        input (Tensor): the input tensor.
        exponent (float or Tensor): the exponent value.
    
    Keyword args:
        out (Tensor, optional): the output tensor.
    
    Returns:
        Tensor: the result tensor.
    """
    # Determine if exponent is a scalar or tensor
    if isinstance(exponent, (int, float)):
        # Scalar exponent case
        y = torch.pow(input, exponent)
    else:
        # Tensor exponent case
        if not isinstance(exponent, Tensor):
            exponent = torch.tensor(exponent, dtype=input.dtype, device=input.device)
        y = torch.pow(input, exponent)
    
    # Handle output tensor if provided
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_pow():
    results = {}

    # Test case 1: input_tensor and exponent are scalars
    input_tensor = torch.tensor([2.0], device='cuda')
    exponent = 3.0
    results["test_case_1"] = pow(input_tensor, exponent)

    # Test case 2: input_tensor is a tensor, exponent is a scalar
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = 2.0
    results["test_case_2"] = pow(input_tensor, exponent)

    # Test case 3: input_tensor and exponent are tensors of the same shape
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    exponent = torch.tensor([3.0, 2.0, 1.0], device='cuda')
    results["test_case_3"] = pow(input_tensor, exponent)

    # Test case 4: input_tensor is a tensor, exponent is a negative scalar
    input_tensor = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    exponent = -0.5
    results["test_case_4"] = pow(input_tensor, exponent)

    return results

test_results = test_pow()
