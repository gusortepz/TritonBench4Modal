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


def ifftshift(input: Tensor, dim=None) -> Tensor:
    """
    Inverse of fftshift. Rearranges elements so the zero-frequency component
    is moved back to the original position (beginning of the tensor).
    
    For each dimension of size n, shifts by ceil(n/2) = (n+1)//2.
    """
    ndim = input.dim()

    if dim is None:
        # Apply to all dimensions
        dims = list(range(ndim))
    elif isinstance(dim, int):
        # Normalize negative dim
        dims = [(dim % ndim)]
    else:
        # Tuple or list of ints
        dims = [d % ndim for d in dim]

    # For ifftshift, shift amount for each dim of size n is (n+1)//2
    shifts = [(input.shape[d] + 1) // 2 for d in dims]

    return torch.roll(input, shifts, dims=dims)

##################################################################################################################################################



import torch

def test_ifftshift():
    results = {}

    # Test case 1: 1D tensor, default dim
    input_tensor_1d = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda')
    results["test_case_1"] = ifftshift(input_tensor_1d)

    # Test case 2: 2D tensor, default dim
    input_tensor_2d = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device='cuda')
    results["test_case_2"] = ifftshift(input_tensor_2d)

    # Test case 3: 2D tensor, specific dim
    results["test_case_3"] = ifftshift(input_tensor_2d, dim=0)

    # Test case 4: 3D tensor, specific dim
    input_tensor_3d = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], device='cuda')
    results["test_case_4"] = ifftshift(input_tensor_3d, dim=(1, 2))

    return results

test_results = test_ifftshift()
