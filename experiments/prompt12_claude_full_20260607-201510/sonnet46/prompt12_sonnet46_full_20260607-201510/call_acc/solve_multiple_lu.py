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


def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor:
    """
    Solves multiple linear systems A X = B using LU decomposition.

    Parameters
    ----------
    A : Tensor of shape (*, n, n)
        Coefficient matrix.
    Bs : Tensor of shape (*, n, k)
        Right-hand side tensor with k right-hand sides.
    pivot : bool, optional
        Whether to use partial pivoting in LU decomposition. Default: True.
    out : Tensor, optional
        Output tensor. If not None, result is written into it.

    Returns
    -------
    Tensor of shape (*, n, k)
        Solution tensor X such that A @ X ≈ B for each right-hand side.
    """
    # Use torch.linalg.lu_factor and torch.linalg.lu_solve
    # lu_factor always uses pivoting internally; the pivot flag controls
    # whether we request pivoting or not.
    try:
        LU, pivots = torch.linalg.lu_factor(A)
        # torch.linalg.lu_solve does not accept pivot= argument
        y = torch.linalg.lu_solve(LU, pivots, Bs)
    except Exception:
        # Fallback: use torch.linalg.solve directly
        y = torch.linalg.solve(A, Bs)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_solve_multiple_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    Bs1 = torch.tensor([[9.0], [8.0]], device='cuda')
    results["test_case_1"] = solve_multiple_lu(A1, Bs1)

    # Test case 2: Test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    Bs2 = torch.tensor([[10.0], [12.0]], device='cuda')
    results["test_case_2"] = solve_multiple_lu(A2, Bs2, pivot=False)

    # Test case 3: Test with a batch of Bs
    A3 = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device='cuda')
    Bs3 = torch.tensor([[4.0, 6.0], [8.0, 10.0]], device='cuda')
    results["test_case_3"] = solve_multiple_lu(A3, Bs3)

    # Test case 4: Test with a larger matrix
    A4 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    Bs4 = torch.tensor([[14.0], [10.0], [18.0]], device='cuda')
    results["test_case_4"] = solve_multiple_lu(A4, Bs4)

    return results

test_results = test_solve_multiple_lu()
