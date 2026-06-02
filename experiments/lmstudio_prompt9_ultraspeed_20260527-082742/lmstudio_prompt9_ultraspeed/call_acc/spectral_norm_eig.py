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

def _spectral_norm_eig_impl(A):
    eigvals = torch.linalg.eigvals(A)
    return torch.amax(torch.abs(eigvals), dim=-1)

try:
    _spectral_norm_eig_fast = torch.compile(_spectral_norm_eig_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _spectral_norm_eig_fast = _spectral_norm_eig_impl

def spectral_norm_eig(A, *, out=None):
    y = _spectral_norm_eig_fast(A)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_spectral_norm_eig():
    results = {}

    # Test case 1: Single 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = spectral_norm_eig(A1)

    # Test case 2: Batch of 2x2 matrices
    A2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_2"] = spectral_norm_eig(A2)

    # Test case 3: Single 3x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_3"] = spectral_norm_eig(A3)

    # Test case 4: Batch of 3x3 matrices
    A4 = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 
                       [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], device='cuda')
    results["test_case_4"] = spectral_norm_eig(A4)

    return results

test_results = test_spectral_norm_eig()
