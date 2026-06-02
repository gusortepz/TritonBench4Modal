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

def _det_impl(A):
    return torch.linalg.det(A)

try:
    _det_fast = torch.compile(_det_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _det_fast = _det_impl

def det(A, *, out=None):
    y = _det_fast(A)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

# def det(A):
#     return torch.linalg.det(A)

def test_det():
    results = {}
    
    # Test case 1: 2x2 identity matrix
    A1 = torch.eye(2, device='cuda')
    results["test_case_1"] = det(A1).item()
    
    # Test case 2: 3x3 matrix with random values
    A2 = torch.rand((3, 3), device='cuda')
    results["test_case_2"] = det(A2).item()
    
    # Test case 3: 4x4 matrix with all zeros
    A3 = torch.zeros((4, 4), device='cuda')
    results["test_case_3"] = det(A3).item()
    
    # Test case 4: 2x2 matrix with specific values
    A4 = torch.tensor([[4.0, 7.0], [2.0, 6.0]], device='cuda')
    results["test_case_4"] = det(A4).item()
    
    return results

test_results = test_det()
