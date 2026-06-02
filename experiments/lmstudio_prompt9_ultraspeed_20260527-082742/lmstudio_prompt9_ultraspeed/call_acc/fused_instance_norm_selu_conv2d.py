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

def _fused_instance_norm_selu_conv2d_impl(input, weight, bias, stride, padding, dilation, groups, num_features, eps, momentum, affine, track_running_stats):
    conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    selu_out = F.selu(conv_out)
    if track_running_stats:
        out_channels = weight.shape[0]
        running_mean = torch.zeros(out_channels, dtype=input.dtype, device=input.device)
        running_var = torch.ones(out_channels, dtype=input.dtype, device=input.device)
    else:
        running_mean = None
        running_var = None
    return F.instance_norm(selu_out, running_mean, running_var, weight=None, bias=None, use_input_stats=True, momentum=momentum, eps=eps)

try:
    _fused_instance_norm_selu_conv2d_fast = torch.compile(_fused_instance_norm_selu_conv2d_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_instance_norm_selu_conv2d_fast = _fused_instance_norm_selu_conv2d_impl

def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False, *, out=None):
    y = _fused_instance_norm_selu_conv2d_fast(input, weight, bias, stride, padding, dilation, groups, num_features, eps, momentum, affine, track_running_stats)
    if out is not None:
        out.copy_(y)
        return out
    return y

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
