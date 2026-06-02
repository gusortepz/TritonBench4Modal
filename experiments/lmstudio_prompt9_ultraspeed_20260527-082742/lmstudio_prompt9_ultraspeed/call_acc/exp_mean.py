import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    y = torch.exp(input).mean(dim=dim, keepdim=keepdim, dtype=dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()
