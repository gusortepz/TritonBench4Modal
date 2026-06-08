import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple, NamedTuple

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


class MaxResult(NamedTuple):
    values: Tensor
    indices: Tensor


def max(
    input: Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    out: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns a namedtuple (values, indices) where values is the maximum value of each row
    of the input tensor in the given dimension dim. Indices is the index location of each
    maximum value found (argmax). If keepdim is True, the output tensors are of the same
    size as input except in the dimension dim where they are of size 1. Otherwise, dim is
    squeezed, resulting in the output tensors having 1 fewer dimension than input.
    If there are multiple maximal values in a reduced row, the indices of the first maximal
    value are returned.
    """
    
    # Normalize dim to positive value
    if dim < 0:
        dim = input.ndim + dim
    
    # Clamp dim to valid range
    if dim < 0 or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim})")
    
    # Use PyTorch's native max implementation
    values, indices = torch.max(input, dim=dim, keepdim=keepdim)
    
    # Handle out parameter
    if out is not None:
        out_values, out_indices = out
        out_values.copy_(values)
        out_indices.copy_(indices)
        return (out_values, out_indices)
    
    return (values, indices)

##################################################################################################################################################



import torch

def test_max():
    results = {}

    # Test case 1: Basic test with a 2D tensor
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_1'] = max(input_tensor, dim=0)

    # Test case 2: Test with keepdim=True
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_2'] = max(input_tensor, dim=1, keepdim=True)

    # Test case 3: Test with a 3D tensor
    input_tensor = torch.tensor([[[1, 3, 2], [4, 6, 5]], [[7, 9, 8], [10, 12, 11]]], device='cuda')
    results['test_case_3'] = max(input_tensor, dim=2)

    # Test case 4: Test with a negative dimension
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_4'] = max(input_tensor, dim=-1)

    return results

test_results = test_max()
