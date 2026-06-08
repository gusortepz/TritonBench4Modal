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


def airy_ai(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the Airy function Ai for each element of the input tensor."""
    y = torch.special.airy_ai(input)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_airy_ai():
    results = {}

    # Test case 1: Single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = airy_ai(input1)

    # Test case 2: Single negative value
    input2 = torch.tensor([-1.0], device='cuda')
    results["test_case_2"] = airy_ai(input2)

    # Test case 3: Tensor with multiple values
    input3 = torch.tensor([0.0, 1.0, -1.0], device='cuda')
    results["test_case_3"] = airy_ai(input3)

    # Test case 4: Tensor with large positive and negative values
    input4 = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_4"] = airy_ai(input4)

    return results

test_results = test_airy_ai()
