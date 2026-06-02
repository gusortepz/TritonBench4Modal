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

def _matrix_power_eig_impl(A, k):
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    Dk = eigenvalues ** k
    V_inv = torch.linalg.inv(eigenvectors)
    return (eigenvectors * Dk.unsqueeze(-1)) @ V_inv

try:
    _matrix_power_eig_fast = torch.compile(_matrix_power_eig_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _matrix_power_eig_fast = _matrix_power_eig_impl

def matrix_power_eig(A, k, *, out=None):
    y = _matrix_power_eig_fast(A, k)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_matrix_power_eig():
    results = {}

    # Test case 1: Simple 2x2 matrix with integer exponent
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    k1 = 2
    results["test_case_1"] = matrix_power_eig(A1, k1)

    # Test case 2: 3x3 matrix with fractional exponent
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    k2 = 0.5
    results["test_case_2"] = matrix_power_eig(A2, k2)

    # Test case 4: Batch of 2x2 matrices with integer exponent
    A4 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    k4 = 3
    results["test_case_4"] = matrix_power_eig(A4, k4)

    return results

test_results = test_matrix_power_eig()
