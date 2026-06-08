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


def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> Tensor:
    """
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes
    the mean value along the specified dimension. If no dimension is specified,
    it computes the mean over all elements.
    """
    # Cast input to dtype if specified
    if dtype is not None:
        x = input.to(dtype=dtype)
    else:
        x = input

    # Handle other: could be a tensor or a number
    if isinstance(other, torch.Tensor):
        if dtype is not None:
            o = other.to(dtype=dtype)
        else:
            o = other
        # Perform scaled addition with broadcasting
        added = torch.add(x, o, alpha=alpha)
    else:
        # other is a number
        added = torch.add(x, alpha * other)

    # Ensure float for mean computation if needed
    # mean requires float or complex input
    if added.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        added = added.to(torch.float64 if dtype == torch.float64 else torch.float32)

    # Compute mean
    if dim is None:
        y = added.mean()
        if keepdim:
            # keepdim with no dim: expand to same ndim as input with all dims=1
            shape = (1,) * input.dim()
            y = y.reshape(shape) if input.dim() > 0 else y
    else:
        y = added.mean(dim=dim, keepdim=keepdim)

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
