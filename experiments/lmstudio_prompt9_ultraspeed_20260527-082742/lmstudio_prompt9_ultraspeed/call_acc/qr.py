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

def qr(A, mode='reduced', *, out=None):
    Q, R = torch.linalg.qr(A, mode=mode)
    if out is not None:
        out[0].copy_(Q)
        out[1].copy_(R)
        return out
    return Q, R

##################################################################################################################################################



import torch

def test_qr():
    results = {}

    # Test case 1: reduced mode, 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    Q1, R1 = qr(A1, mode='reduced')
    results["test_case_1"] = (Q1.cpu(), R1.cpu())

    # Test case 2: complete mode, 3x2 matrix
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    Q2, R2 = qr(A2, mode='complete')
    results["test_case_2"] = (Q2.cpu(), R2.cpu())

    # Test case 3: r mode, 2x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    Q3, R3 = qr(A3, mode='r')
    results["test_case_3"] = (Q3.cpu(), R3.cpu())

    # Test case 4: reduced mode, batch of 2x2 matrices
    A4 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    Q4, R4 = qr(A4, mode='reduced')
    results["test_case_4"] = (Q4.cpu(), R4.cpu())

    return results

test_results = test_qr()
