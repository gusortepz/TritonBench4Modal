import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Perform batch matrix multiplication
    bmm_result = torch.bmm(input1, input2)
    
    # Apply RMS normalization
    if isinstance(normalized_shape, int):
        norm_dims = (-1,)
        normalized_shape_tuple = (normalized_shape,)
    else:
        normalized_shape_tuple = tuple(normalized_shape)
        norm_dims = tuple(range(-len(normalized_shape_tuple), 0))
    
    # Compute RMS normalization manually to ensure correctness
    var = bmm_result.float().pow(2).mean(dim=norm_dims, keepdim=True)
    rms_norm_result = (bmm_result.float() * torch.rsqrt(var + eps)).to(bmm_result.dtype)
    
    # Apply GELU activation
    if approximate == 'none':
        gelu_result = torch.nn.functional.gelu(rms_norm_result)
    elif approximate == 'tanh':
        gelu_result = torch.nn.functional.gelu(rms_norm_result, approximate='tanh')
    else:
        gelu_result = torch.nn.functional.gelu(rms_norm_result)
    
    # Apply dropout
    if training:
        dropout_result = F.dropout(gelu_result, p=dropout_p, training=True)
    else:
        dropout_result = gelu_result
    
    # Subtract the third tensor
    y = dropout_result - other
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    return y

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
