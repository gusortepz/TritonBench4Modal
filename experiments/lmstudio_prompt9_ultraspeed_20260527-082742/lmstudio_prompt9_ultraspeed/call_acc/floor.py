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

def floor(input, *, out=None):
    y = torch.floor(input)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_floor():
    results = {}

    # Test case 1: Simple tensor with positive and negative floats
    input1 = torch.tensor([1.7, -2.3, 3.5, -4.8], device='cuda')
    results["test_case_1"] = floor(input1)

    # Test case 2: Tensor with integers (should remain unchanged)
    input2 = torch.tensor([1, -2, 3, -4], device='cuda')
    results["test_case_2"] = floor(input2)

    # Test case 3: Tensor with zero and positive/negative floats
    input3 = torch.tensor([0.0, 2.9, -3.1, 4.0], device='cuda')
    results["test_case_3"] = floor(input3)

    # Test case 4: Large tensor with random floats
    input4 = torch.rand(1000, device='cuda') * 100 - 50  # Random floats between -50 and 50
    results["test_case_4"] = floor(input4)

    return results

test_results = test_floor()
