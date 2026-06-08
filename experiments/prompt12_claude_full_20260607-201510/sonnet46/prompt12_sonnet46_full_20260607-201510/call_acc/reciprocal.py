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


def reciprocal(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    # Promote integral types to default float type
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                       torch.uint8, torch.bool):
        promoted = input.to(torch.get_default_dtype())
    else:
        promoted = input

    y = torch.reciprocal(promoted)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_reciprocal():
    results = {}

    # Test case 1: Basic test with positive and negative numbers
    a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298], device='cuda')
    results["test_case_1"] = reciprocal(a)

    # Test case 2: Test with a tensor containing zero (expecting inf)
    b = torch.tensor([0.0, 1.0, -1.0, 2.0], device='cuda')
    results["test_case_2"] = reciprocal(b)

    # Test case 3: Test with a tensor containing large numbers
    c = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_3"] = reciprocal(c)

    # Test case 4: Test with a tensor of ones (expecting ones)
    d = torch.ones(4, device='cuda')
    results["test_case_4"] = reciprocal(d)

    return results

test_results = test_reciprocal()
