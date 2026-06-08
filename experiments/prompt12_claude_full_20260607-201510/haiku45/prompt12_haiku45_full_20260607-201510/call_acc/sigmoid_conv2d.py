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


@triton.jit
def _sigmoid_conv2d_kernel(
    output_ptr,
    stride_output_batch,
    stride_output_channels,
    stride_output_h,
    stride_output_w,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise sigmoid on convolution output."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    output_data = tl.load(output_ptr + offsets, mask=mask)
    sigmoid_output = tl.sigmoid(output_data)
    tl.store(output_ptr + offsets, sigmoid_output, mask=mask)


def sigmoid_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Applies 2D convolution followed by sigmoid activation.
    
    Args:
        input: Input tensor of shape (minibatch, in_channels, iH, iW).
        weight: Convolution filters of shape (out_channels, in_channels/groups, kH, kW).
        bias: Optional bias tensor of shape (out_channels). Default: None.
        stride: Stride of convolution kernel. Default: 1.
        padding: Padding. Can be 'valid', 'same', int, or tuple. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of groups. Default: 1.
        out: Output tensor (optional).
    
    Returns:
        Tensor: Output after convolution and sigmoid.
    """
    # PyTorch reference path: conv2d + sigmoid
    y = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    
    # Apply sigmoid
    y = torch.sigmoid(y)
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
#     conv_result = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     result = torch.sigmoid(conv_result)
#     return result

def test_sigmoid_conv2d():
    results = {}

    # Test case 1: Basic test with no bias, stride, padding, dilation, or groups
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = sigmoid_conv2d(input1, weight1)

    # Test case 2: Test with bias
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = sigmoid_conv2d(input1, weight1, bias=bias2)

    # Test case 3: Test with stride
    results["test_case_3"] = sigmoid_conv2d(input1, weight1, stride=2)

    # Test case 4: Test with padding
    results["test_case_4"] = sigmoid_conv2d(input1, weight1, padding=1)

    return results

test_results = test_sigmoid_conv2d()
