import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union


def _sigmoid_argmax_impl(input, dim=None, keepdim=False):
    s = torch.sigmoid(input)
    return torch.argmax(s, dim=dim, keepdim=keepdim)


try:
    _sigmoid_argmax_compiled = torch.compile(_sigmoid_argmax_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _sigmoid_argmax_compiled = _sigmoid_argmax_impl


def sigmoid_argmax(input, dim=None, keepdim=False):
    y = _sigmoid_argmax_compiled(input, dim, keepdim)
    return y

##################################################################################################################################################



import torch

def test_sigmoid_argmax():
    results = {}

    # Test case 1: 1D tensor, no dim specified
    input1 = torch.tensor([0.1, 2.0, -1.0, 3.0], device='cuda')
    results["test_case_1"] = sigmoid_argmax(input1)

    # Test case 2: 2D tensor, dim=0
    input2 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_2"] = sigmoid_argmax(input2, dim=0)

    # Test case 3: 2D tensor, dim=1
    input3 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_3"] = sigmoid_argmax(input3, dim=1)

    # Test case 4: 2D tensor, dim=1, keepdim=True
    input4 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_4"] = sigmoid_argmax(input4, dim=1, keepdim=True)

    return results

test_results = test_sigmoid_argmax()
