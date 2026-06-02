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

def _mul_sub_impl(input, other_mul, other_sub, alpha):
    return input * other_mul - alpha * other_sub

try:
    _mul_sub_fast = torch.compile(_mul_sub_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _mul_sub_fast = _mul_sub_impl

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> torch.Tensor:
    y = _mul_sub_fast(input, other_mul, other_sub, alpha)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_mul_sub():
    results = {}

    # Test case 1: input, other_mul, other_sub are tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_mul_tensor = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    other_sub_tensor = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = mul_sub(input_tensor, other_mul_tensor, other_sub_tensor)

    # Test case 2: input is a tensor, other_mul is a number, other_sub is a tensor
    other_mul_number = 2.0
    results["test_case_2"] = mul_sub(input_tensor, other_mul_number, other_sub_tensor)

    # Test case 3: input is a tensor, other_mul is a tensor, other_sub is a number
    other_sub_number = 0.5
    results["test_case_3"] = mul_sub(input_tensor, other_mul_tensor, other_sub_number)

    # Test case 4: input, other_mul, other_sub are numbers
    input_number = 3.0
    results["test_case_4"] = mul_sub(input_number, other_mul_number, other_sub_number)

    return results

test_results = test_mul_sub()
