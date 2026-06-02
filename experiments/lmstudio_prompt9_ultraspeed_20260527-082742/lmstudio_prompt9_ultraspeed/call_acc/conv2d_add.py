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

def _conv2d_add_impl(input, weight, bias, other, stride, padding, dilation, groups, alpha):
    y = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if other is not None:
        y = y + other * alpha
    return y

try:
    _conv2d_add_fast = torch.compile(_conv2d_add_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _conv2d_add_fast = _conv2d_add_impl

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    y = _conv2d_add_fast(input, weight, bias, other, stride, padding, dilation, groups, alpha)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



def test_conv2d_add():
    results = {}

    # Test case 1: Basic convolution with bias, no addition
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(2, device='cuda')
    results["test_case_1"] = conv2d_add(input_tensor, weight_tensor, bias=bias_tensor)

    # Test case 2: Convolution with addition of a scalar
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    scalar_addition = 2.0
    results["test_case_2"] = conv2d_add(input_tensor, weight_tensor, other=scalar_addition)

    # Test case 3: Convolution with addition of a tensor
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    results["test_case_3"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor)

    # Test case 4: Convolution with addition of a tensor and alpha scaling
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    alpha_value = 0.5
    results["test_case_4"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor, alpha=alpha_value)

    return results

test_results = test_conv2d_add()
