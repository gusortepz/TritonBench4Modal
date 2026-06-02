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

def _determinant_lu_impl(A, pivot):
    if pivot:
        LU, pivots = torch.linalg.lu_factor(A)
        n = LU.shape[-1]
        diag = torch.diagonal(LU, dim1=-2, dim2=-1)
        idx = torch.arange(1, n + 1, device=pivots.device, dtype=torch.int64)
        swaps = (pivots != idx).sum(dim=-1)
        sign = (-1.0) ** swaps
        return sign * torch.prod(diag, dim=-1)
    else:
        return torch.linalg.det(A)

try:
    _determinant_lu_fast = torch.compile(_determinant_lu_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _determinant_lu_fast = _determinant_lu_impl

def determinant_lu(A, *, pivot=True, out=None):
    y = _determinant_lu_fast(A, pivot)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_determinant_lu():
    results = {}

    # Test case 1: 2x2 matrix with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [2.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_lu(A1)

    # Test case 2: 3x3 matrix with pivot=False
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    results["test_case_2"] = determinant_lu(A2, pivot=False)

    # Test case 3: Batch of 2x2 matrices with pivot=True
    A3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = determinant_lu(A3)

    # Test case 4: 4x4 matrix with pivot=True
    A4 = torch.tensor([[1.0, 0.0, 2.0, -1.0],
                       [3.0, 0.0, 0.0, 5.0],
                       [2.0, 1.0, 4.0, -3.0],
                       [1.0, 0.0, 5.0, 0.0]], device='cuda')
    results["test_case_4"] = determinant_lu(A4)

    return results

test_results = test_determinant_lu()
