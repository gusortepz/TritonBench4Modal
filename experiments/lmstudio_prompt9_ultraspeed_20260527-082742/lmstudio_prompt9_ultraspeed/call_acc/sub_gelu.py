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

def _sub_gelu_impl(input, other, alpha, approximate):
    return F.gelu(input - alpha * other, approximate=approximate)

try:
    _sub_gelu_fast = torch.compile(_sub_gelu_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _sub_gelu_fast = _sub_gelu_impl

def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    y = _sub_gelu_fast(input, other, alpha, approximate)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_sub_gelu():
    results = {}

    # Test case 1: Basic subtraction and GELU with default approximate
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub_gelu(input_tensor, other_tensor)

    # Test case 2: Subtraction with alpha and GELU with default approximate
    alpha = 0.5
    results["test_case_2"] = sub_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Subtraction and GELU with 'tanh' approximation
    approximate = 'tanh'
    results["test_case_3"] = sub_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Subtraction with alpha and GELU with 'tanh' approximation
    results["test_case_4"] = sub_gelu(input_tensor, other_tensor, alpha=alpha, approximate=approximate)

    return results

test_results = test_sub_gelu()
