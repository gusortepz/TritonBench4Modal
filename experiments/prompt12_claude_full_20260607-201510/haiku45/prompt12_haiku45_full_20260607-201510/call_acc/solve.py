import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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


def solve(A: Tensor, B: Tensor, *, left: bool = True, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the solution of a square system of linear equations with a unique solution.
    Supports inputs of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices.
    Assumes that matrix A is invertible.
    
    Solves AX = B if left=True, or XA = B if left=False.
    
    Args:
        A: Square matrix or batch of square matrices of shape (..., n, n)
        B: Matrix or batch of matrices of shape (..., n, k) if left=True, or (..., k, n) if left=False
        left: If True, solve AX = B (default). If False, solve XA = B.
        out: Optional output tensor to store the result
    
    Returns:
        Solution X with same shape as B
    """
    
    # Use torch.linalg.solve for both left and right solve
    if left:
        # Solve AX = B
        y = torch.linalg.solve(A, B)
    else:
        # Solve XA = B
        # Transpose: X @ A = B  =>  A^T @ X^T = B^T
        A_T = A.mT
        B_T = B.mT
        X_T = torch.linalg.solve(A_T, B_T)
        y = X_T.mT
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

# def solve(A, B):
#     return torch.linalg.solve(A, B)

def test_solve():
    results = {}

    # Test case 1: Simple 2x2 system
    A1 = torch.tensor([[3.0, 1.0], [1.0, 2.0]], device='cuda')
    B1 = torch.tensor([9.0, 8.0], device='cuda')
    results["test_case_1"] = solve(A1, B1)

    # Test case 2: Larger 3x3 system
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    B2 = torch.tensor([6.0, 4.0, 3.0], device='cuda')
    results["test_case_2"] = solve(A2, B2)

    # Test case 3: Singular matrix (should raise an error)
    try:
        A3 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
        B3 = torch.tensor([5.0, 10.0], device='cuda')
        results["test_case_3"] = solve(A3, B3)
    except RuntimeError as e:
        results["test_case_3"] = str(e)

    # Test case 4: Non-square matrix (should raise an error)
    try:
        A4 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
        B4 = torch.tensor([7.0, 8.0], device='cuda')
        results["test_case_4"] = solve(A4, B4)
    except RuntimeError as e:
        results["test_case_4"] = str(e)

    return results

test_results = test_solve()
