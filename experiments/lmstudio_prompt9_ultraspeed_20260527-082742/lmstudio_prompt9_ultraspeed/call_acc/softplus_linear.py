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

def _softplus_linear_impl(input, weight, bias, beta, threshold):
    return F.softplus(F.linear(input, weight, bias), beta=beta, threshold=threshold)

try:
    _softplus_linear_fast = torch.compile(_softplus_linear_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _softplus_linear_fast = _softplus_linear_impl

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    return _softplus_linear_fast(input, weight, bias, beta, threshold)

##################################################################################################################################################



import torch

def test_softplus_linear():
    results = {}

    # Test case 1: Basic test with default beta and threshold
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = softplus_linear(input1, weight1, bias1)

    # Test case 2: Test with non-default beta
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias2 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_2"] = softplus_linear(input2, weight2, bias2, beta=2)

    # Test case 3: Test with non-default threshold
    input3 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device='cuda')
    weight3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = softplus_linear(input3, weight3, bias3, threshold=15)

    # Test case 4: Test with no bias
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = softplus_linear(input4, weight4)

    return results

test_results = test_softplus_linear()
