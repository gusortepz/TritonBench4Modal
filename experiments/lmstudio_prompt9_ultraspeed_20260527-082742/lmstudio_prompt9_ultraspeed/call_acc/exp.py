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

def exp(input, *, out=None):
    y = torch.exp(input)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_exp():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp(input_tensor_1)

    # Test case 2: Test with a tensor containing negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp(input_tensor_2)

    # Test case 3: Test with a tensor containing zero
    input_tensor_3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp(input_tensor_3)

    # Test case 4: Test with a larger tensor
    input_tensor_4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = exp(input_tensor_4)

    return results

test_results = test_exp()
