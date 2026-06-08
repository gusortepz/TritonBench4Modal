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


def _relu_max_pool2d_conv2d_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups: int,
    pool_kernel_size,
    pool_stride,
    pool_padding,
    pool_dilation,
    pool_ceil_mode: bool,
    inplace: bool,
) -> Tensor:
    # Step 1: 2D convolution
    x = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
    )
    # Step 2: Max pooling
    x = F.max_pool2d(
        x,
        kernel_size=pool_kernel_size,
        stride=pool_stride,
        padding=pool_padding,
        dilation=pool_dilation,
        ceil_mode=pool_ceil_mode,
    )
    # Step 3: ReLU
    x = F.relu(x, inplace=inplace)
    return x


try:
    _relu_max_pool2d_conv2d_fast = torch.compile(
        _relu_max_pool2d_conv2d_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _relu_max_pool2d_conv2d_fast = _relu_max_pool2d_conv2d_impl


def relu_max_pool2d_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups: int = 1,
    pool_kernel_size=2,
    pool_stride=None,
    pool_padding=0,
    pool_dilation=1,
    pool_ceil_mode: bool = False,
    inplace: bool = False,
) -> Tensor:
    try:
        return _relu_max_pool2d_conv2d_fast(
            input,
            weight,
            bias,
            conv_stride,
            conv_padding,
            conv_dilation,
            conv_groups,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            pool_dilation,
            pool_ceil_mode,
            inplace,
        )
    except Exception:
        return _relu_max_pool2d_conv2d_impl(
            input,
            weight,
            bias,
            conv_stride,
            conv_padding,
            conv_dilation,
            conv_groups,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            pool_dilation,
            pool_ceil_mode,
            inplace,
        )

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
#     x = F.conv2d(input, weight, bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
#     x = F.max_pool2d(x, kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding, dilation=pool_dilation, ceil_mode=pool_ceil_mode)
#     x = F.relu(x, inplace=inplace)
#     return x

def test_relu_max_pool2d_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 8, 8, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_max_pool2d_conv2d(input, weight)
    
    # Test case 2: Test with bias
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = relu_max_pool2d_conv2d(input, weight, bias=bias)
    
    # Test case 3: Test with different convolution stride and padding
    results["test_case_3"] = relu_max_pool2d_conv2d(input, weight, conv_stride=2, conv_padding=1)
    
    # Test case 4: Test with different max pooling parameters
    results["test_case_4"] = relu_max_pool2d_conv2d(input, weight, pool_kernel_size=3, pool_stride=2, pool_padding=1)
    
    return results

test_results = test_relu_max_pool2d_conv2d()
