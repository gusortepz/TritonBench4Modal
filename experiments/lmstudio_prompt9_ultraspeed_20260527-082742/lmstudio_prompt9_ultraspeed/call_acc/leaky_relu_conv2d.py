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

def _leaky_relu_conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    y = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    y = F.leaky_relu(y, negative_slope=negative_slope, inplace=inplace)
    return y

try:
    _leaky_relu_conv2d_fast = torch.compile(_leaky_relu_conv2d_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _leaky_relu_conv2d_fast = _leaky_relu_conv2d_impl

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False, *, out=None):
    y = _leaky_relu_conv2d_fast(input, weight, bias, stride, padding, dilation, groups, negative_slope, inplace)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def leaky_relu_conv2d(input: Tensor, weight: Tensor, bias: Tensor=None, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, negative_slope: float=0.01, inplace: bool=False) -> Tensor:
#     conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     output = F.leaky_relu(conv_output, negative_slope, inplace)
#     return output

def test_leaky_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_1"] = leaky_relu_conv2d(input, weight, bias)
    
    # Test case 2: Test with stride and padding
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = leaky_relu_conv2d(input, weight, bias, stride=2, padding=1)
    
    # Test case 3: Test with dilation and groups
    input = torch.randn(1, 6, 32, 32, device='cuda')
    weight = torch.randn(6, 1, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_3"] = leaky_relu_conv2d(input, weight, bias, dilation=2, groups=6)
    
    # Test case 4: Test with negative_slope and inplace
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_4"] = leaky_relu_conv2d(input, weight, bias, negative_slope=0.1, inplace=True)
    
    return results

test_results = test_leaky_relu_conv2d()
