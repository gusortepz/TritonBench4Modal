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


def sub(input: Tensor, other: Union[Tensor, float], *, alpha: float = 1, out: Optional[Tensor] = None) -> Tensor:
    """
    Subtracts other, scaled by alpha, from input.
    out = input - alpha * other
    """
    if isinstance(other, (int, float, complex)):
        y = torch.sub(input, other, alpha=alpha)
    else:
        y = torch.sub(input, other, alpha=alpha)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_sub():
    results = {}

    # Test case 1: Basic subtraction with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub(input1, other1)

    # Test case 2: Subtraction with alpha
    input2 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    other2 = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    results["test_case_2"] = sub(input2, other2, alpha=2)

    # Test case 3: Subtraction with a scalar other
    input3 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    other3 = 2.0
    results["test_case_3"] = sub(input3, other3)

    # Test case 4: Subtraction with out parameter
    input4 = torch.tensor([10.0, 11.0, 12.0], device='cuda')
    other4 = torch.tensor([3.0, 3.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = sub(input4, other4, out=out4)

    return results

test_results = test_sub()
