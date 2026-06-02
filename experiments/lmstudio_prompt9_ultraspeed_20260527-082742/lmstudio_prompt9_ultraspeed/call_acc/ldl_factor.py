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


def ldl_factor(A, *, hermitian=False, out=None):
    if out is not None:
        LD, pivots = torch.linalg.ldl_factor(A, hermitian=hermitian)
        out[0].copy_(LD)
        out[1].copy_(pivots)
        return out
    return torch.linalg.ldl_factor(A, hermitian=hermitian)

##################################################################################################################################################



import torch

def test_ldl_factor():
    results = {}

    # Test case 1: Symmetric matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    results["test_case_1"] = ldl_factor(A1)

    # Test case 2: Hermitian matrix
    A2 = torch.tensor([[2.0, 1.0j], [-1.0j, 2.0]], device='cuda')
    results["test_case_2"] = ldl_factor(A2, hermitian=True)

    # Test case 3: Batch of symmetric matrices
    A3 = torch.tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.5], [0.5, 2.0]]], device='cuda')
    results["test_case_3"] = ldl_factor(A3)

    # Test case 4: Batch of Hermitian matrices
    A4 = torch.tensor([[[2.0, 1.0j], [-1.0j, 2.0]], [[3.0, 0.5j], [-0.5j, 3.0]]], device='cuda')
    results["test_case_4"] = ldl_factor(A4, hermitian=True)

    return results

test_results = test_ldl_factor()
