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

def _fused_transformer_block_impl(input, weight1, weight2, residual, dropout_p, eps):
    z1 = torch.matmul(input, weight1)
    z2 = F.softmax(z1, dim=-1)
    z3 = F.dropout(z2, p=dropout_p, training=True)
    z4 = torch.matmul(z3, weight2)
    z5 = z4 + residual
    y = F.layer_norm(z5, z5.shape[-1:], eps=eps)
    return y

try:
    _fused_transformer_block_fast = torch.compile(_fused_transformer_block_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_transformer_block_fast = _fused_transformer_block_impl

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    y = _fused_transformer_block_fast(input, weight1, weight2, residual, dropout_p, eps)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_transformer_block():
    results = {}

    # Test case 1: Basic functionality test
    input1 = torch.randn(2, 3, 4, device='cuda')
    weight1_1 = torch.randn(4, 5, device='cuda')
    weight2_1 = torch.randn(5, 4, device='cuda')
    residual1 = torch.randn(2, 3, 4, device='cuda')
    results["test_case_1"] = fused_transformer_block(input1, weight1_1, weight2_1, residual1)

    # Test case 2: Different input size
    input2 = torch.randn(1, 5, 6, device='cuda')
    weight1_2 = torch.randn(6, 7, device='cuda')
    weight2_2 = torch.randn(7, 6, device='cuda')
    residual2 = torch.randn(1, 5, 6, device='cuda')
    results["test_case_2"] = fused_transformer_block(input2, weight1_2, weight2_2, residual2)

    # Test case 3: Test with dropout probability set to 0
    input3 = torch.randn(3, 2, 4, device='cuda')
    weight1_3 = torch.randn(4, 5, device='cuda')
    weight2_3 = torch.randn(5, 4, device='cuda')
    residual3 = torch.randn(3, 2, 4, device='cuda')
    results["test_case_3"] = fused_transformer_block(input3, weight1_3, weight2_3, residual3, dropout_p=0.0)

    # Test case 4: Test with a different epsilon value
    input4 = torch.randn(4, 3, 5, device='cuda')
    weight1_4 = torch.randn(5, 6, device='cuda')
    weight2_4 = torch.randn(6, 5, device='cuda')
    residual4 = torch.randn(4, 3, 5, device='cuda')
    results["test_case_4"] = fused_transformer_block(input4, weight1_4, weight2_4, residual4, eps=1e-3)

    return results

test_results = test_fused_transformer_block()
