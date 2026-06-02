import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

def _fused_index_select_eq_impl(input, dim, index, other):
    return torch.eq(torch.index_select(input, dim, index), other)

try:
    _fused_index_select_eq_compiled = torch.compile(_fused_index_select_eq_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_index_select_eq_compiled = _fused_index_select_eq_impl

def fused_index_select_eq(input, dim, index, other, *, out=None):
    y = _fused_index_select_eq_compiled(input, dim, index, other)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()
