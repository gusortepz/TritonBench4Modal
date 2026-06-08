import torch
import torch.nn.functional as F
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


def ifftshift(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Inverse of torch.fft.fftshift. Rearranges elements of input tensor from FFT order
    such that the zero-frequency component is moved back to the original position.
    
    Args:
        input (Tensor): the tensor in FFT order
        dim (int, Tuple[int], optional): The dimensions to rearrange.
            Only dimensions specified here will be rearranged, any other dimensions
            will be left in their original order.
            Default: All dimensions of input.
    
    Returns:
        Tensor: The rearranged tensor with zero-frequency component at original position.
    """
    
    if dim is None:
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)
    
    shape = input.shape
    
    for d in dim:
        d = d % input.ndim
        n = shape[d]
        mid = (n + 1) // 2
        input = torch.roll(input, shifts=mid, dims=d)
    
    return input

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
