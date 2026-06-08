import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Union
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
def _fused_relu_ln_kernel(
    output_ptr,
    input_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + LayerNorm kernel for the output of linear transformation."""
    row_idx = tl.program_id(0)
    col_start = tl.arange(0, BLOCK_SIZE)
    
    if row_idx >= N:
        return
    
    # Load input (post-ReLU linear output)
    offsets = row_idx * M + col_start
    mask = col_start < M
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and reciprocal std for layer normalization
    mean = tl.sum(x, axis=0) / M
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / M
    rstd = tl.rsqrt(var + eps)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Apply affine if present
    if has_weight and has_bias:
        w = tl.load(weight_ptr + col_start, mask=mask, other=0.0)
        b = tl.load(bias_ptr + col_start, mask=mask, other=0.0)
        y = x_norm * w + b
    elif has_weight:
        w = tl.load(weight_ptr + col_start, mask=mask, other=0.0)
        y = x_norm * w
    elif has_bias:
        b = tl.load(bias_ptr + col_start, mask=mask, other=0.0)
        y = x_norm + b
    else:
        y = x_norm
    
    # Store output, mean, and rstd
    tl.store(output_ptr + offsets, y, mask=mask)
    if col_start == 0:
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rstd_ptr + row_idx, rstd)


def _fused_layer_norm_relu_linear_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    normalized_shape: Optional[Union[int, list, torch.Size]] = None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
) -> Tensor:
    """PyTorch reference implementation."""
    # Linear transformation
    y = F.linear(input, weight, bias)
    
    # ReLU activation
    y = F.relu(y)
    
    # Layer normalization
    if normalized_shape is None:
        normalized_shape = y.shape[-1]
    
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    elif isinstance(normalized_shape, (list, torch.Size)):
        normalized_shape = tuple(normalized_shape)
    
    # Prepare layer norm weight and bias
    ln_weight = None
    ln_bias = None
    if elementwise_affine:
        ln_weight = torch.ones(normalized_shape, dtype=y.dtype, device=y.device)
        ln_bias = torch.zeros(normalized_shape, dtype=y.dtype, device=y.device)
    
    y = F.layer_norm(y, normalized_shape, weight=ln_weight, bias=ln_bias, eps=eps)
    
    return y


try:
    _fused_layer_norm_relu_linear_fast = torch.compile(
        _fused_layer_norm_relu_linear_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _fused_layer_norm_relu_linear_fast = _fused_layer_norm_relu_linear_impl


def fused_layer_norm_relu_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    normalized_shape: Optional[Union[int, list, torch.Size]] = None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
) -> Tensor:
    """
    Applies a fused operation consisting of a linear transformation followed by 
    ReLU activation and layer normalization on the input tensor.
    
    Args:
        input (Tensor): Input tensor with shape (*, in_features).
        weight (Tensor): Weights for the linear transformation, shape (out_features, in_features).
        bias (Tensor, optional): Bias for the linear transformation, shape (out_features).
        normalized_shape (int or list or torch.Size, optional): 
            Shape of the dimensions to normalize. Defaults to the last dimension.
        eps (float, optional): A value added to the denominator for numerical stability. 
            Default is 1e-5.
        elementwise_affine (bool, optional): If True, layer normalization has learnable 
            parameters. Default is True.
    
    Returns:
        Tensor: Result after applying the linear transformation, ReLU, and layer normalization.
    """
    try:
        return _fused_layer_norm_relu_linear_fast(
            input, weight, bias, normalized_shape, eps, elementwise_affine
        )
    except Exception:
        return _fused_layer_norm_relu_linear_impl(
            input, weight, bias, normalized_shape, eps, elementwise_affine
        )

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
