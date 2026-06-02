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

def lu(A, *, pivot=True, out=None):
    P, L, U = torch.linalg.lu(A, pivot=pivot)
    if out is not None:
        out[0].copy_(P)
        out[1].copy_(L)
        out[2].copy_(U)
        return out
    return (P, L, U)

##################################################################################################################################################



import torch

def test_lu():
    results = {}

    # Test case 1: 2x2 matrix with pivoting
    A1 = torch.randn(2, 2, device="cuda")
    P1, L1, U1 = lu(A1)
    results["test_case_1"] = (P1.cpu(), L1.cpu(), U1.cpu())

    # Test case 2: 3x3 matrix with pivoting
    A2 = torch.randn(3, 3, device="cuda")
    P2, L2, U2 = lu(A2)
    results["test_case_2"] = (P2.cpu(), L2.cpu(), U2.cpu())

    # Test case 3: 2x3 matrix without pivoting
    A3 = torch.randn(2, 3, device="cuda")
    P3, L3, U3 = lu(A3, pivot=False)
    results["test_case_3"] = (P3.cpu(), L3.cpu(), U3.cpu())

    # Test case 4: Batch of 2x2 matrices with pivoting
    A4 = torch.randn(4, 2, 2, device="cuda")
    P4, L4, U4 = lu(A4)
    results["test_case_4"] = (P4.cpu(), L4.cpu(), U4.cpu())

    return results

test_results = test_lu()
