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


def bessel_j1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    r"""
    Computes the Bessel function of the first kind of order 1 for each element
    of the input tensor.
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword Args:
        out (Tensor, optional): the output tensor. Default: None
    
    Returns:
        Tensor: the output tensor.
    """
    # Use torch.special.bessel_j1 for the computation
    # This is the standard PyTorch implementation for Bessel J1
    y = torch.special.bessel_j1(input)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_bessel_j1():
    results = {}

    # Test case 1: Basic test with a single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = bessel_j1(input1)

    # Test case 2: Test with a tensor of multiple values
    input2 = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    results["test_case_2"] = bessel_j1(input2)

    # Test case 3: Test with a tensor of negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = bessel_j1(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.linspace(-5.0, 5.0, steps=10, device='cuda')
    results["test_case_4"] = bessel_j1(input4)

    return results

test_results = test_bessel_j1()
