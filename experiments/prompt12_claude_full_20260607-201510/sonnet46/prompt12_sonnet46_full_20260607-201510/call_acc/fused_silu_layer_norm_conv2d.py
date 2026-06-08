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


def _fused_silu_layer_norm_conv2d_impl(
    x: Tensor,
    weight: Tensor,
    conv_weight: Tensor,
    conv_bias: Optional[Tensor],
    conv_stride: int,
    conv_padding: int,
    conv_dilation: int,
    conv_groups: int,
    ln_eps: float,
) -> Tensor:
    # Step 1: 2D Convolution
    y = F.conv2d(x, conv_weight, conv_bias,
                 stride=conv_stride, padding=conv_padding,
                 dilation=conv_dilation, groups=conv_groups)
    # y shape: (N, C_out, H_out, W_out)
    # Step 2: Layer Normalization
    # weight shape is (C_out,), so normalized_shape = (C_out,)
    # For (N, C, H, W) tensor with normalized_shape=(C,), we need to permute
    # to (N, H, W, C), apply LN, then permute back
    N, C, H, W = y.shape
    if weight is not None:
        norm_shape = tuple(weight.shape)
    else:
        norm_shape = (C,)
    
    # Check if normalized_shape matches last dims of y
    # If norm_shape == (C,) but y's last dim is W, we need to permute
    if norm_shape == (C,) and W != C:
        # Permute to (N, H, W, C), normalize, permute back
        y_perm = y.permute(0, 2, 3, 1).contiguous()
        w = weight if weight is not None and tuple(weight.shape) == norm_shape else None
        y_norm = F.layer_norm(y_perm, norm_shape, weight=w, bias=None, eps=ln_eps)
        y = y_norm.permute(0, 3, 1, 2).contiguous()
    else:
        # Try to apply directly
        w = weight if weight is not None and tuple(weight.shape) == norm_shape else None
        y = F.layer_norm(y, norm_shape, weight=w, bias=None, eps=ln_eps)
    
    # Step 3: SiLU activation
    y = F.silu(y)
    return y


try:
    _fused_silu_layer_norm_conv2d_fast = torch.compile(
        _fused_silu_layer_norm_conv2d_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _fused_silu_layer_norm_conv2d_fast = _fused_silu_layer_norm_conv2d_impl


def fused_silu_layer_norm_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor = None,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    conv_groups: int = 1,
    ln_eps: float = 1e-5,
) -> torch.Tensor:
    try:
        return _fused_silu_layer_norm_conv2d_fast(
            x, weight, conv_weight, conv_bias,
            conv_stride, conv_padding, conv_dilation, conv_groups, ln_eps,
        )
    except Exception:
        return _fused_silu_layer_norm_conv2d_impl(
            x, weight, conv_weight, conv_bias,
            conv_stride, conv_padding, conv_dilation, conv_groups, ln_eps,
        )

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_silu_layer_norm_conv2d(x: torch.Tensor, weight: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor=None, conv_stride: int=1, conv_padding: int=0, conv_dilation: int=1, conv_groups: int=1, ln_eps: float=1e-05) -> torch.Tensor:
#     conv_out = F.conv2d(x, conv_weight, bias=conv_bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
#     normalized_out = F.layer_norm(conv_out, conv_out.shape[1:], eps=ln_eps)
#     output = F.silu(normalized_out)
#     return output

def test_fused_silu_layer_norm_conv2d():
    results = {}
    
    # Test case 1: Basic functionality with default parameters
    x = torch.randn(1, 3, 5, 5, device='cuda')
    conv_weight = torch.randn(6, 3, 3, 3, device='cuda')
    results['test_case_1'] = fused_silu_layer_norm_conv2d(x, None, conv_weight)
    
    # Test case 2: With conv_bias
    conv_bias = torch.randn(6, device='cuda')
    results['test_case_2'] = fused_silu_layer_norm_conv2d(x, None, conv_weight, conv_bias=conv_bias)
    
    # Test case 3: With different stride and padding
    results['test_case_3'] = fused_silu_layer_norm_conv2d(x, None, conv_weight, conv_stride=2, conv_padding=1)
    
    # Test case 4: With different dilation and groups
    results['test_case_4'] = fused_silu_layer_norm_conv2d(x, None, conv_weight, conv_dilation=2, conv_groups=1)
    
    return results

test_results = test_fused_silu_layer_norm_conv2d()
