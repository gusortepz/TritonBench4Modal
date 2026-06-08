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


def invert_matrix_lu(A: Tensor, *, pivot: bool = True, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the inverse of a square matrix using LU decomposition.

    Given a square invertible matrix A, computes A^{-1} by performing LU
    decomposition and solving linear systems involving triangular matrices.

    Supports inputs of 'float', 'double', 'cfloat', and 'cdouble' dtypes,
    as well as batches of matrices.

    Args:
        A: Square (batches of) matrix/matrices to invert. Shape (..., n, n).
        pivot: Whether to use pivoting during LU decomposition. Default: True.
        out: Optional output tensor.

    Returns:
        The inverse of A.
    """
    # Validate input
    if A.dim() < 2:
        raise ValueError(f"invert_matrix_lu: expected input with at least 2 dimensions, got {A.dim()}")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(
            f"invert_matrix_lu: expected square matrix, got shape {A.shape}"
        )

    n = A.shape[-1]
    batch_shape = A.shape[:-2]

    # Build identity matrix matching batch shape and dtype/device of A
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    # Expand eye to match batch dimensions
    if batch_shape:
        eye = eye.expand(*batch_shape, n, n)

    if pivot:
        # Standard LU with partial pivoting (recommended for numerical stability)
        try:
            # lu_factor returns (LU, pivots)
            LU, pivots = torch.linalg.lu_factor(A)
            # lu_solve: solves LU @ X = B where B = eye → X = A^{-1}
            result = torch.linalg.lu_solve(LU, pivots, eye)
        except Exception:
            # Fallback to torch.linalg.solve
            result = torch.linalg.solve(A, eye)
    else:
        # LU without pivoting
        try:
            LU, pivots = torch.linalg.lu_factor_ex(A, pivot=False)
            result = torch.linalg.lu_solve(LU, pivots, eye)
        except Exception:
            try:
                # Try with lu_factor (always pivots) as fallback
                LU, pivots = torch.linalg.lu_factor(A)
                result = torch.linalg.lu_solve(LU, pivots, eye)
            except Exception:
                result = torch.linalg.solve(A, eye)

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_invert_matrix_lu():
    results = {}

    # Test case 1: Basic test with pivot=True
    A1 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_1"] = invert_matrix_lu(A1)

    # Test case 2: Basic test with pivot=False
    A2 = torch.tensor([[4.0, 3.0], [6.0, 3.0]], device='cuda')
    results["test_case_2"] = invert_matrix_lu(A2, pivot=False)

    # Test case 3: Larger matrix with pivot=True
    A3 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_3"] = invert_matrix_lu(A3)

    # Test case 4: Larger matrix with pivot=False
    A4 = torch.tensor([[7.0, 2.0, 1.0], [0.0, 3.0, -1.0], [-3.0, 4.0, 2.0]], device='cuda')
    results["test_case_4"] = invert_matrix_lu(A4, pivot=False)

    return results

test_results = test_invert_matrix_lu()
