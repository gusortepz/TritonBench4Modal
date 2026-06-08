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


def permute_copy(input: Tensor, dims) -> Tensor:
    """
    Performs the same operation as torch.permute, which rearranges the dimensions
    of the input tensor according to the specified dims, but all output tensors are
    freshly created instead of aliasing the input.
    """
    # torch.permute returns a view; we need a fresh copy (no aliasing)
    return torch.permute(input, dims).contiguous()

##################################################################################################################################################



import torch

def test_permute_copy():
    results = {}

    # Test case 1: Simple 2D tensor permutation
    tensor_2d = torch.tensor([[1, 2], [3, 4]], device='cuda')
    results["test_case_1"] = permute_copy(tensor_2d, [1, 0])

    # Test case 2: 3D tensor permutation
    tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device='cuda')
    results["test_case_2"] = permute_copy(tensor_3d, [2, 0, 1])

    # Test case 3: Permutation with no change
    tensor_no_change = torch.tensor([1, 2, 3, 4], device='cuda')
    results["test_case_3"] = permute_copy(tensor_no_change, [0])

    # Test case 4: Higher dimensional tensor permutation
    tensor_4d = torch.rand((2, 3, 4, 5), device='cuda')
    results["test_case_4"] = permute_copy(tensor_4d, [3, 2, 1, 0])

    return results

test_results = test_permute_copy()
