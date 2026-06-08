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


def polygamma(n: int, input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the n-th derivative of the digamma function.
    
    Args:
        n (int): the order of the polygamma function (n >= 0)
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: the polygamma values
    """
    y = torch.polygamma(n, input)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

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
