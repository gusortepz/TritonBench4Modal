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

def _gammaln_impl(input):
    return torch.special.gammaln(input)

try:
    _gammaln_fast = torch.compile(_gammaln_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _gammaln_fast = _gammaln_impl

def gammaln(input, *, out=None):
    y = _gammaln_fast(input)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

# def gammaln(input: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
#     """
#     Computes the natural logarithm of the absolute value of the gamma function on the input tensor.
    
#     Args:
#         input (torch.Tensor): the input tensor.
#         out (torch.Tensor, optional): the output tensor.

#     Returns:
#         torch.Tensor: tensor containing the natural log of the gamma function for each element in the input.
#     """
#     return torch.special.gammaln(input, out=out)

def test_gammaln():
    results = {}
    
    # Test case 1: Single value tensor
    input1 = torch.tensor([2.0], device='cuda')
    results["test_case_1"] = gammaln(input1)
    
    # Test case 2: Multi-value tensor
    input2 = torch.tensor([2.0, 3.0, 4.0], device='cuda')
    results["test_case_2"] = gammaln(input2)
    
    # Test case 3: Tensor with negative values
    input3 = torch.tensor([-2.5, -3.5, -4.5], device='cuda')
    results["test_case_3"] = gammaln(input3)
    
    # Test case 4: Large tensor
    input4 = torch.tensor([i for i in range(1, 1001)], dtype=torch.float32, device='cuda')
    results["test_case_4"] = gammaln(input4)
    
    return results

test_results = test_gammaln()
