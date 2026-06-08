import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl
from typing import Optional, Union, Tuple

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


def fused_instance_norm_selu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    num_features: Optional[int] = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False,
) -> Tensor:
    """
    Fused operation: 2D convolution -> SELU -> instance normalization.
    
    Args:
        input: Input tensor of shape (minibatch, in_channels, iH, iW).
        weight: Weights for the convolution, shape (out_channels, in_channels / groups, kH, kW).
        bias: Bias for the convolution layer, shape (out_channels). Optional.
        stride: Stride of the convolution. Default is 1.
        padding: Padding for the convolution. Default is 0.
        dilation: Spacing between kernel elements. Default is 1.
        groups: Number of blocked connections from input channels to output channels. Default is 1.
        num_features: Number of features or channels for instance normalization.
        eps: A value added to the denominator for numerical stability. Default is 1e-5.
        momentum: Momentum for updating running statistics. Default is 0.1.
        affine: If True, instance normalization has learnable affine parameters. Default is False.
        track_running_stats: If True, tracks running mean and variance. Default is False.
    
    Returns:
        Output tensor after conv2d, SELU, and instance normalization.
    """
    # Step 1: Apply 2D convolution
    conv_out = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    
    # Step 2: Apply SELU activation
    selu_out = F.selu(conv_out)
    
    # Step 3: Apply instance normalization
    # Infer num_features from conv_out if not provided
    if num_features is None:
        num_features = conv_out.shape[1]  # out_channels
    
    # Instance normalization: normalize over spatial dimensions (H, W)
    # PyTorch's instance_norm normalizes over all dims except batch and channel
    instance_norm_out = F.instance_norm(
        selu_out,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        momentum=momentum,
        eps=eps,
    )
    
    return instance_norm_out

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import nn

# def fused_instance_norm_selu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) -> torch.Tensor:
#     conv_output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     selu_output = torch.nn.functional.selu(conv_output)
#     normalized_output = torch.nn.functional.instance_norm(selu_output, eps=eps, momentum=momentum)
#     return normalized_output

def test_fused_instance_norm_selu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(3, 3, 3, 3, device='cuda')
    results["test_case_1"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor)
    
    # Test case 2: Test with stride
    results["test_case_2"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, stride=2)
    
    # Test case 3: Test with padding
    results["test_case_3"] = fused_instance_norm_selu_conv2d(input_tensor, weight_tensor, padding=1)
    
    return results

test_results = test_fused_instance_norm_selu_conv2d()
