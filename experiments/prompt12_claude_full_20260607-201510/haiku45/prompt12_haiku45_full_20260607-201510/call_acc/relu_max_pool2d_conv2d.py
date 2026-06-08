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


def relu_max_pool2d_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    conv_stride: Union[int, Tuple[int, int]] = 1,
    conv_padding: Union[int, Tuple[int, int], str] = 0,
    conv_dilation: Union[int, Tuple[int, int]] = 1,
    conv_groups: int = 1,
    pool_kernel_size: Union[int, Tuple[int, int]] = 2,
    pool_stride: Optional[Union[int, Tuple[int, int]]] = None,
    pool_padding: Union[int, Tuple[int, int]] = 0,
    pool_dilation: Union[int, Tuple[int, int]] = 1,
    pool_ceil_mode: bool = False,
    inplace: bool = False,
) -> Tensor:
    """
    Applies 2D convolution, max pooling, and ReLU activation in sequence.
    
    Args:
        input: Input tensor of shape (minibatch, in_channels, iH, iW)
        weight: Convolution filters of shape (out_channels, in_channels / groups, kH, kW)
        bias: Optional bias tensor of shape (out_channels)
        conv_stride: Stride of the convolution kernel
        conv_padding: Padding added to input in convolution
        conv_dilation: Spacing between kernel elements in convolution
        conv_groups: Number of blocked connections from input to output channels
        pool_kernel_size: Size of the pooling region
        pool_stride: Stride of the pooling operation (defaults to pool_kernel_size)
        pool_padding: Padding added to input in max pooling
        pool_dilation: Stride between elements within a sliding window
        pool_ceil_mode: If True, uses ceil instead of floor for output shape
        inplace: If True, performs ReLU in-place
    
    Returns:
        Output tensor after conv2d, max_pool2d, and relu operations
    """
    # Step 1: Apply 2D convolution
    conv_output = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
    )
    
    # Step 2: Apply max pooling
    if pool_stride is None:
        pool_stride = pool_kernel_size
    
    pool_output = F.max_pool2d(
        conv_output,
        kernel_size=pool_kernel_size,
        stride=pool_stride,
        padding=pool_padding,
        dilation=pool_dilation,
        ceil_mode=pool_ceil_mode,
    )
    
    # Step 3: Apply ReLU activation
    if inplace:
        return F.relu(pool_output, inplace=True)
    else:
        return F.relu(pool_output, inplace=False)

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
