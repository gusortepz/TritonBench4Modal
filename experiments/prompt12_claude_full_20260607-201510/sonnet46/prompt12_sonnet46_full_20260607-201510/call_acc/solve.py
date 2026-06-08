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


def solve(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes the solution of a square system of linear equations with a unique solution.
    
    If left=True (default), solves A @ X = B.
    If left=False, solves X @ A = B.
    
    Supports float, double, cfloat, cdouble dtypes.
    Supports batches of matrices.
    Assumes A is invertible.
    """
    if left:
        y = torch.linalg.solve(A, B, left=True)
    else:
        y = torch.linalg.solve(A, B, left=False)

    if out is not None:
        out.resize_as_(y)
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
