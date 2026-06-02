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

def _add_gelu_impl(input, other, alpha, approximate):
    return F.gelu(input + alpha * other, approximate=approximate)

try:
    _add_gelu_fast = torch.compile(_add_gelu_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _add_gelu_fast = _add_gelu_impl

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    y = _add_gelu_fast(input, other, alpha, approximate)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_add_gelu():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_gelu(input_tensor, other_tensor)

    # Test case 2: Test with alpha parameter
    alpha = 2
    results["test_case_2"] = add_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Test with approximate='tanh'
    approximate = 'tanh'
    results["test_case_3"] = add_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Test with a scalar 'other'
    other_scalar = 0.5
    results["test_case_4"] = add_gelu(input_tensor, other_scalar)

    return results

test_results = test_add_gelu()
