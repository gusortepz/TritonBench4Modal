import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from torch import Tensor

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


def fused_qr_solve(A: Tensor, b: Tensor) -> Tensor:
    """
    Solves the linear system Ax = b using QR decomposition.
    
    Given matrix A of shape (m, n) where m >= n, and right-hand side b of shape (m, k),
    computes x = R^{-1} (Q^T b).
    
    Args:
        A: The matrix A of shape (m, n) where m >= n.
        b: The right-hand side tensor b of shape (m, k).
    
    Returns:
        x: The solution tensor of shape (n, k).
    """
    Q, R = torch.linalg.qr(A, mode='reduced')
    # Q^T b
    Qt_b = Q.mT @ b  # shape (n, k)
    # Solve R x = Q^T b using triangular solve
    x = torch.linalg.solve_triangular(R, Qt_b, upper=True)
    return x

##################################################################################################################################################



import torch

def test_fused_qr_solve():
    results = {}

    # Test case 1: Square matrix A and vector b
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b1 = torch.tensor([[5.0], [6.0]], device='cuda')
    results["test_case_1"] = fused_qr_solve(A1, b1)

    # Test case 2: Rectangular matrix A (m > n) and vector b
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b2 = torch.tensor([[7.0], [8.0], [9.0]], device='cuda')
    results["test_case_2"] = fused_qr_solve(A2, b2)

    # Test case 3: Square matrix A and matrix b with multiple columns
    A3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = fused_qr_solve(A3, b3)

    # Test case 4: Rectangular matrix A (m > n) and matrix b with multiple columns
    A4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b4 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    results["test_case_4"] = fused_qr_solve(A4, b4)

    return results

test_results = test_fused_qr_solve()
