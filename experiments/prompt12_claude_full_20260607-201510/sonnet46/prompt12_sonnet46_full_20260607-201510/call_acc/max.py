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


def max(input: Tensor, dim: int, keepdim: bool = False, *, out=None) -> Tuple[Tensor, Tensor]:
    """
    Returns a namedtuple (values, indices) where values is the maximum value
    of each row of the input tensor in the given dimension dim.
    Indices is the index location of each maximum value found (argmax).
    If keepdim is True, the output tensors are of the same size as input
    except in the dimension dim where they are of size 1.
    If there are multiple maximal values in a reduced row, the indices of
    the first maximal value are returned.
    """
    result = torch.max(input, dim=dim, keepdim=keepdim)
    values = result.values
    indices = result.indices

    if out is not None:
        if isinstance(out, (tuple, list)) and len(out) == 2:
            out[0].copy_(values)
            out[1].copy_(indices)
            return type(result)(out[0], out[1])
        
    return result

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
