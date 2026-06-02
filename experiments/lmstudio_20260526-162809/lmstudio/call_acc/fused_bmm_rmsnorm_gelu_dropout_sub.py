import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union


def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None) -> torch.Tensor:
    # Batch matrix multiplication: (B, N, M) @ (B, M, P) -> (B, N, P)
    result = torch.bmm(input1, input2)

    # Handle normalized_shape: int, list, or torch.Size
    if isinstance(normalized_shape, int):
        normalized_shape_tuple = (normalized_shape,)
    else:
        normalized_shape_tuple = tuple(normalized_shape)

    # RMS normalization over the last len(normalized_shape) dimensions
    try:
        result = F.rms_norm(result, normalized_shape_tuple, eps=eps)
    except AttributeError:
        # Fallback for older PyTorch versions without F.rms_norm
        norm_dims = tuple(range(-len(normalized_shape_tuple), 0))
        var = result.float().pow(2).mean(dim=norm_dims, keepdim=True)
        result = (result.float() * torch.rsqrt(var + eps)).to(result.dtype)

    # GELU activation
    result = F.gelu(result, approximate=approximate)

    # Dropout (respects training mode)
    result = F.dropout(result, p=dropout_p, training=training)

    # Subtract other tensor (broadcastable to output shape)
    result = result - other

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_bmm_rmsnorm_gelu_dropout_sub():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    other = torch.randn(2, 3, 5, device='cuda')
    normalized_shape = 5
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)

    # Test case 2: Test with different dropout probability
    dropout_p = 0.3
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=dropout_p)

    # Test case 3: Test with training set to False
    training = False
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, training=training)

    # Test case 4: Test with approximate GELU
    approximate = 'tanh'
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, approximate=approximate)

    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout_sub()
