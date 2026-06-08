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


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> Tensor:
    """
    Applies a 2D convolution over an input image composed of several input planes.
    
    Args:
        input: input tensor of shape (minibatch, in_channels, iH, iW)
        weight: filters of shape (out_channels, in_channels/groups, kH, kW)
        bias: optional bias tensor of shape (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
        padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'}, 
                 single number or a tuple (padH, padW). Default: 0
        dilation: the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1
        groups: split input into groups, both in_channels and out_channels should be divisible 
                by the number of groups. Default: 1
    
    Returns:
        output tensor
    """
    return F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

##################################################################################################################################################



import torch

def test_conv2d():
    results = {}

    # Test case 1: Basic convolution with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = conv2d(input1, weight1)

    # Test case 2: Convolution with stride
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_2"] = conv2d(input2, weight2, stride=2)

    # Test case 3: Convolution with padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = conv2d(input3, weight3, padding=1)

    # Test case 4: Convolution with dilation
    input4 = torch.randn(1, 3, 5, 5, device='cuda')
    weight4 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_4"] = conv2d(input4, weight4, dilation=2)

    return results

test_results = test_conv2d()
