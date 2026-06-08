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


def zeta(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the Hurwitz zeta function elementwise.
    
    zeta(x, q) = sum_{n=0}^{inf} 1 / (n + q)^x
    
    Args:
        input (Tensor): the input tensor corresponding to `x`.
        other (Tensor): the input tensor corresponding to `q`.
        out (Tensor, optional): the output tensor.
    
    Returns:
        Tensor: result of Hurwitz zeta function applied elementwise.
    """
    if out is not None:
        return torch.special.zeta(input, other, out=out)
    return torch.special.zeta(input, other)

##################################################################################################################################################



import torch

def test_zeta():
    results = {}

    # Test case 1: Basic test with simple values
    input1 = torch.tensor([2.0, 3.0], device='cuda')
    other1 = torch.tensor([1.0, 2.0], device='cuda')
    results["test_case_1"] = zeta(input1, other1)

    # Test case 2: Test with larger values
    input2 = torch.tensor([10.0, 20.0], device='cuda')
    other2 = torch.tensor([5.0, 10.0], device='cuda')
    results["test_case_2"] = zeta(input2, other2)

    # Test case 3: Test with fractional values
    input3 = torch.tensor([2.5, 3.5], device='cuda')
    other3 = torch.tensor([1.5, 2.5], device='cuda')
    results["test_case_3"] = zeta(input3, other3)

    # Test case 4: Test with negative values
    input4 = torch.tensor([-2.0, -3.0], device='cuda')
    other4 = torch.tensor([1.0, 2.0], device='cuda')
    results["test_case_4"] = zeta(input4, other4)

    return results

test_results = test_zeta()
