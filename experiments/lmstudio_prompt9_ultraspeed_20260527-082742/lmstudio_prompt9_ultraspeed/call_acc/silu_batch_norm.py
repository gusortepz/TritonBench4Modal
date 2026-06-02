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

def _silu_batch_norm_impl(input, running_mean, running_var, weight, bias, training, momentum, eps):
    out = F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    return F.silu(out)

try:
    _silu_batch_norm_fast = torch.compile(_silu_batch_norm_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _silu_batch_norm_fast = _silu_batch_norm_impl

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5) -> torch.Tensor:
    return _silu_batch_norm_fast(input, running_mean, running_var, weight, bias, training, momentum, eps)

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_silu_batch_norm():
    results = {}

    # Test case 1: Basic functionality with training=False
    input_tensor = torch.randn(3, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = silu_batch_norm(input_tensor, running_mean, running_var, training=False)

    # Test case 2: With weight and bias, training=False
    weight = torch.ones(5, device='cuda')
    bias = torch.zeros(5, device='cuda')
    results["test_case_2"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=False)

    # Test case 3: With training=True
    results["test_case_3"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True)

    # Test case 4: Different momentum and eps values
    results["test_case_4"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=0.2, eps=1e-3)

    return results

test_results = test_silu_batch_norm()
