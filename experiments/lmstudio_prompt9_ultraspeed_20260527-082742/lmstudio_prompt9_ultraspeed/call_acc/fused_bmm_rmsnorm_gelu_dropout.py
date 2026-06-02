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

def _fused_bmm_rmsnorm_gelu_dropout_impl(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none'):
    z = torch.bmm(input1, input2)
    if isinstance(normalized_shape, int):
        normalized_shape_tuple = (normalized_shape,)
    else:
        normalized_shape_tuple = tuple(normalized_shape)
    y = F.rms_norm(z, normalized_shape_tuple, weight=None, eps=eps)
    y = F.gelu(y, approximate=approximate)
    y = F.dropout(y, p=dropout_p, training=training)
    return y

try:
    _fused_bmm_rmsnorm_gelu_dropout_fast = torch.compile(_fused_bmm_rmsnorm_gelu_dropout_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_bmm_rmsnorm_gelu_dropout_fast = _fused_bmm_rmsnorm_gelu_dropout_impl

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    y = _fused_bmm_rmsnorm_gelu_dropout_fast(input1, input2, normalized_shape, dropout_p, eps, training, approximate)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-05, training=True, approximate='none', *, out=None):
#     z1 = torch.bmm(input1, input2)
#     rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
#     gelu_out = F.gelu(rms_norm, approximate=approximate)
#     output = F.dropout(gelu_out, p=dropout_p, training=training)
#     if out is not None:
#         out.copy_(output)
#         return out
#     return output

def test_fused_bmm_rmsnorm_gelu_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5)
    
    # Test case 2: Different dropout probability
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, dropout_p=0.2)
    
    # Test case 3: Non-training mode
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, training=False)
    
    # Test case 4: Different approximation method for GELU
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, approximate='tanh')
    
    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout()
