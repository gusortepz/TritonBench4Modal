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


def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    upscale_factor=2,
) -> torch.Tensor:
    # Apply 2D convolution
    conv_out = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    # Apply pixel shuffle to rearrange spatial dimensions
    out = F.pixel_shuffle(conv_out, upscale_factor)
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
#     x = F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.pixel_shuffle(x, upscale_factor)

def test_pixel_shuffle_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 4, 8, 8, device='cuda')
    weight1 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_1"] = pixel_shuffle_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 4, 8, 8, device='cuda')
    weight2 = torch.randn(16, 4, 3, 3, device='cuda')
    bias2 = torch.randn(16, device='cuda')
    results["test_case_2"] = pixel_shuffle_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride
    input3 = torch.randn(1, 4, 16, 16, device='cuda')
    weight3 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_3"] = pixel_shuffle_conv2d(input3, weight3, stride=2)
    
    # Test case 4: Test with padding
    input4 = torch.randn(1, 4, 8, 8, device='cuda')
    weight4 = torch.randn(16, 4, 3, 3, device='cuda')
    results["test_case_4"] = pixel_shuffle_conv2d(input4, weight4, padding=1)
    
    return results

test_results = test_pixel_shuffle_conv2d()
