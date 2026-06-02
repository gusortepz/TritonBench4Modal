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

def _fused_layer_norm_relu_linear_impl(input, weight, bias, normalized_shape, eps, elementwise_affine):
    z = F.linear(input, weight, bias)
    z = F.relu(z)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ln_weight = torch.ones(normalized_shape, device=input.device, dtype=input.dtype) if elementwise_affine else None
    ln_bias = torch.zeros(normalized_shape, device=input.device, dtype=input.dtype) if elementwise_affine else None
    return F.layer_norm(z, normalized_shape, weight=ln_weight, bias=ln_bias, eps=eps)

try:
    _fused_layer_norm_relu_linear_fast = torch.compile(_fused_layer_norm_relu_linear_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_layer_norm_relu_linear_fast = _fused_layer_norm_relu_linear_impl

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True, *, out=None):
    y = _fused_layer_norm_relu_linear_fast(input, weight, bias, normalized_shape, eps, elementwise_affine)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn as nn

def test_fused_layer_norm_relu_linear():
    results = {}

    # Test case 1: Basic test with bias
    input1 = torch.randn(4, 5, device='cuda')
    weight1 = torch.randn(3, 5, device='cuda')
    bias1 = torch.randn(3, device='cuda')
    normalized_shape1 = 3
    results["test_case_1"] = fused_layer_norm_relu_linear(input1, weight1, bias1, normalized_shape1)

    # Test case 2: Without bias
    input2 = torch.randn(4, 5, device='cuda')
    weight2 = torch.randn(3, 5, device='cuda')
    normalized_shape2 = 3
    results["test_case_2"] = fused_layer_norm_relu_linear(input2, weight2, None, normalized_shape2)

    # Test case 3: Different normalized shape
    input3 = torch.randn(4, 5, device='cuda')
    weight3 = torch.randn(3, 5, device='cuda')
    bias3 = torch.randn(3, device='cuda')
    normalized_shape3 = torch.Size([3])
    results["test_case_3"] = fused_layer_norm_relu_linear(input3, weight3, bias3, normalized_shape3)

    # Test case 4: Different epsilon value
    input4 = torch.randn(4, 5, device='cuda')
    weight4 = torch.randn(3, 5, device='cuda')
    bias4 = torch.randn(3, device='cuda')
    normalized_shape4 = 3
    eps4 = 1e-3
    results["test_case_4"] = fused_layer_norm_relu_linear(input4, weight4, bias4, normalized_shape4, eps=eps4)

    return results

test_results = test_fused_layer_norm_relu_linear()
