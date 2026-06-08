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


def polygamma(n, input, *, out=None) -> Tensor:
    """
    Computes the n-th derivative of the digamma function on input.
    Implemented for nonnegative integers n >= 0.

    Args:
        n (int): the order of the polygamma function
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor

    Returns:
        Tensor: result of the polygamma function
    """
    if out is not None:
        return torch.polygamma(n, input, out=out)
    return torch.polygamma(n, input)

##################################################################################################################################################



import torch

def test_polygamma():
    results = {}

    # Test case 1: Basic functionality with n=1
    a = torch.tensor([1, 0.5], device='cuda')
    results["test_case_1"] = polygamma(1, a)

    # Test case 2: Basic functionality with n=2
    results["test_case_2"] = polygamma(2, a)

    # Test case 3: Basic functionality with n=3
    results["test_case_3"] = polygamma(3, a)

    # Test case 4: Basic functionality with n=4
    results["test_case_4"] = polygamma(4, a)

    return results

test_results = test_polygamma()
