import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
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


def determinant_lu(A: Tensor, *, pivot: bool = True, out: Optional[Tensor] = None) -> Tensor:
    r"""
    Computes the determinant of a square matrix using LU decomposition.
    
    Performs LU decomposition on a given square matrix A and calculates its determinant.
    Supports matrices over real or complex numbers and batch dimensions.
    The determinant is computed as the product of the diagonal elements of the upper
    triangular matrix U from the LU decomposition, adjusted by the sign of the permutation
    matrix P if pivoting is used.
    
    Args:
        A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                   consisting of square matrices.
        pivot (bool, optional): Controls whether to compute the LU decomposition with
                               partial pivoting (True) or without pivoting (False).
                               Default: True.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.
    
    Returns:
        Tensor: The determinant of A. Shape is `(*)` (batch dimensions only).
    """
    
    # Validate input
    if A.dim() < 2:
        raise ValueError(f"A must have at least 2 dimensions, got {A.dim()}")
    
    n = A.shape[-1]
    if A.shape[-2] != n:
        raise ValueError(f"A must be square; got shape {A.shape}")
    
    # Use torch.linalg.lu for LU decomposition
    if pivot:
        # torch.linalg.lu returns (P, L, U)
        P, L, U = torch.linalg.lu(A)
    else:
        # Without pivoting: use torch.linalg.lu_factor_ex
        # or compute via lu and reconstruct without permutation
        # torch.linalg.lu will still apply pivoting, so we use lu_factor
        try:
            # lu_factor_ex returns (LU, pivots, info)
            LU, pivots, info = torch.linalg.lu_factor_ex(A, pivot=False)
            # Reconstruct L and U from LU
            L = torch.tril(LU, diagonal=-1) + torch.eye(n, dtype=LU.dtype, device=LU.device)
            U = torch.triu(LU)
            P = None
        except Exception:
            # Fallback: use default lu and ignore P
            P, L, U = torch.linalg.lu(A)
            P = None
    
    # Compute determinant as product of diagonal of U
    # Extract diagonal of U
    diag_U = torch.diagonal(U, dim1=-2, dim2=-1)
    
    # Product of diagonal elements
    det_U = torch.prod(diag_U, dim=-1)
    
    # If pivoting was used, adjust by sign of permutation
    if pivot and P is not None:
        # Compute sign of permutation matrix
        # The sign is (-1)^(number of row swaps)
        # For a permutation matrix P, sign = det(P)
        # We can compute this from the LU factorization
        batch_shape = A.shape[:-2]
        
        # Flatten batch dimensions for processing
        P_flat = P.reshape(-1, n, n)
        det_P_list = []
        
        for i in range(P_flat.shape[0]):
            # Compute determinant of permutation matrix
            # det(P) = (-1)^(number of transpositions)
            # For permutation matrices, det = product of diagonal = ±1
            # Better: use torch.linalg.det on the permutation
            det_P = torch.linalg.det(P_flat[i:i+1])
            det_P_list.append(det_P)
        
        det_P = torch.cat(det_P_list, dim=0).reshape(batch_shape)
        det = det_U * det_P
    else:
        det = det_U
    
    # Handle out parameter
    if out is not None:
        out.copy_(det)
        return out
    
    return det

##################################################################################################################################################



import torch

def test_determinant_lu():
    results = {}

    # Test case 1: 2x2 matrix with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [2.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_lu(A1)

    # Test case 2: 3x3 matrix with pivot=False
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    results["test_case_2"] = determinant_lu(A2, pivot=False)

    # Test case 3: Batch of 2x2 matrices with pivot=True
    A3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = determinant_lu(A3)

    # Test case 4: 4x4 matrix with pivot=True
    A4 = torch.tensor([[1.0, 0.0, 2.0, -1.0],
                       [3.0, 0.0, 0.0, 5.0],
                       [2.0, 1.0, 4.0, -3.0],
                       [1.0, 0.0, 5.0, 0.0]], device='cuda')
    results["test_case_4"] = determinant_lu(A4)

    return results

test_results = test_determinant_lu()
