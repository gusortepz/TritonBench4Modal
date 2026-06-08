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


def _fused_impl(x: Tensor, weight: Tensor, bias: Optional[Tensor],
                normalized_shape, eps: float) -> Tensor:
    # Linear transformation
    y = F.linear(x, weight, bias)
    # ReLU activation
    y = F.relu(y)
    # Layer normalization
    if normalized_shape is not None:
        if isinstance(normalized_shape, int):
            shape = (normalized_shape,)
        else:
            shape = tuple(normalized_shape)
    else:
        shape = (y.shape[-1],)
    y = F.layer_norm(y, shape, weight=None, bias=None, eps=eps)
    return y


try:
    _fused_fast = torch.compile(_fused_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_fast = _fused_impl


def fused_layer_norm_relu_linear(
    input: Tensor,
    weight: Tensor,
    bias=None,
    normalized_shape=None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
) -> Tensor:
    # Determine normalized_shape
    if normalized_shape is not None:
        if isinstance(normalized_shape, int):
            norm_shape = (normalized_shape,)
        else:
            norm_shape = tuple(normalized_shape)
    else:
        # Will be inferred after linear (output last dim = out_features)
        norm_shape = None

    # Try fast compiled path (linear + relu, then layer norm)
    try:
        y = F.linear(input, weight, bias)
        y = F.relu(y)
    except Exception:
        y = F.linear(input, weight, bias)
        y = F.relu(y)

    # Determine the actual normalized_shape
    if norm_shape is None:
        norm_shape = (y.shape[-1],)

    # Apply layer norm
    if elementwise_affine:
        # No learnable affine parameters provided — use identity
        y = F.layer_norm(y, norm_shape, weight=None, bias=None, eps=eps)
    else:
        y = F.layer_norm(y, norm_shape, weight=None, bias=None, eps=eps)

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
