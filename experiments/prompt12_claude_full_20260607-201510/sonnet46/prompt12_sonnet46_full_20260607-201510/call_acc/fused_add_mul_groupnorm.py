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


def _fused_add_mul_groupnorm_impl(
    input1: Tensor,
    input2: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> Tensor:
    # Step 1: element-wise addition
    added = input1 + input2
    # Step 2: element-wise multiplication with input2
    multiplied = added * input2
    # Step 3: group normalization
    # F.group_norm expects (N, C, *) shape
    # weight shape: (C,), bias shape: (C,)
    result = F.group_norm(multiplied, num_groups, weight=weight, bias=bias, eps=eps)
    return result


try:
    _fused_fast = torch.compile(
        _fused_add_mul_groupnorm_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _fused_fast = _fused_add_mul_groupnorm_impl


def fused_add_mul_groupnorm(
    input1: Tensor,
    input2: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: int,
    eps: float = 1e-5,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    try:
        y = _fused_fast(input1, input2, weight, bias, num_groups, eps)
    except Exception:
        y = _fused_add_mul_groupnorm_impl(input1, input2, weight, bias, num_groups, eps)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_add_mul_groupnorm():
    results = {}

    # Test case 1: Basic functionality test
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 2
    results["test_case_1"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 2: Different shapes for input1 and input2 (broadcastable)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(1, 4, 1, 1, device='cuda')  # Broadcastable shape
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 2
    results["test_case_2"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 3: Single group normalization (equivalent to layer normalization)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    weight = torch.randn(4, device='cuda')
    bias = torch.randn(4, device='cuda')
    num_groups = 1
    results["test_case_3"] = fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups)

    # Test case 4: No weight and bias (should default to None)
    input1 = torch.randn(2, 4, 4, 4, device='cuda')
    input2 = torch.randn(2, 4, 4, 4, device='cuda')
    num_groups = 2
    results["test_case_4"] = fused_add_mul_groupnorm(input1, input2, None, None, num_groups)

    return results

test_results = test_fused_add_mul_groupnorm()
