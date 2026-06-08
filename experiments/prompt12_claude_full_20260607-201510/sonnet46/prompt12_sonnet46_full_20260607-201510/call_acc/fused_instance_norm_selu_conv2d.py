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


def _fused_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    num_features: Optional[int],
    eps: float,
    affine: bool,
) -> Tensor:
    # Step 1: 2D convolution
    x = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding,
                 dilation=dilation, groups=groups)
    
    # Step 2: SELU activation
    x = F.selu(x)
    
    # Step 3: Instance normalization
    # num_features for instance_norm is the number of channels (C dimension)
    # instance_norm expects (N, C, H, W) - no learnable params when affine=False
    x = F.instance_norm(x, running_mean=None, running_var=None,
                        weight=None, bias=None, use_input_stats=True, eps=eps)
    
    return x


try:
    _fused_fast = torch.compile(_fused_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_fast = _fused_impl


def fused_instance_norm_selu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    num_features=None,
    eps=1e-5,
    momentum=0.1,
    affine=False,
    track_running_stats=False,
) -> Tensor:
    # Validate inputs
    if not input.is_cuda:
        # Direct PyTorch path for CPU
        x = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding,
                     dilation=dilation, groups=groups)
        x = F.selu(x)
        x = F.instance_norm(x, running_mean=None, running_var=None,
                            weight=None, bias=None, use_input_stats=True, eps=eps)
        return x
    
    try:
        return _fused_fast(input, weight, bias, stride, padding, dilation,
                           groups, num_features, eps, affine)
    except Exception:
        return _fused_impl(input, weight, bias, stride, padding, dilation,
                           groups, num_features, eps, affine)

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
