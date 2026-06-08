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


def low_rank_svd_approximation(A: Tensor, k: int, *, full_matrices: bool = True, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes a rank-k approximation of matrix A using SVD.

    A_k = U[:, :k] @ diag(S[:k]) @ Vh[:k, :]

    Args:
        A (Tensor): Input tensor of shape (*, m, n).
        k (int): Rank of the approximation.
        full_matrices (bool): If True, computes full U and Vh matrices. Default: True.
        out (Tensor, optional): Output tensor. Default: None.

    Returns:
        Tensor: Low-rank approximation of A with same shape as A.
    """
    # Validate k
    if A.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {A.dim()}")

    m = A.shape[-2]
    n = A.shape[-1]
    min_mn = min(m, n)

    if k < 1 or k > min_mn:
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n) = {min_mn}, got k={k}")

    # Compute SVD
    # U: (*, m, m) if full_matrices else (*, m, min(m,n))
    # S: (*, min(m,n))
    # Vh: (*, n, n) if full_matrices else (*, min(m,n), n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    # Slice to keep only top-k components
    # U_k: (*, m, k)
    U_k = U[..., :, :k]
    # S_k: (*, k)
    S_k = S[..., :k]
    # Vh_k: (*, k, n)
    Vh_k = Vh[..., :k, :]

    # Reconstruct: A_k = U_k @ diag(S_k) @ Vh_k
    # Use broadcasting for the diagonal multiplication
    # S_k[..., :, None] expands S_k for broadcasting with Vh_k
    # Result shape: (*, m, n)
    A_k = torch.matmul(U_k * S_k.unsqueeze(-2), Vh_k)

    if out is not None:
        out.copy_(A_k)
        return out
    return A_k

##################################################################################################################################################



import torch

def test_low_rank_svd_approximation():
    results = {}

    # Test case 1: Basic rank-k approximation with full_matrices=True
    A = torch.randn(5, 4, device='cuda')
    k = 2
    results["test_case_1"] = low_rank_svd_approximation(A, k)

    # Test case 2: Basic rank-k approximation with full_matrices=False
    A = torch.randn(6, 3, device='cuda')
    k = 2
    results["test_case_2"] = low_rank_svd_approximation(A, k, full_matrices=False)

    # Test case 3: Batch matrix with full_matrices=True
    A = torch.randn(2, 5, 4, device='cuda')
    k = 3
    results["test_case_3"] = low_rank_svd_approximation(A, k)

    # Test case 4: Batch matrix with full_matrices=False
    A = torch.randn(3, 6, 3, device='cuda')
    k = 2
    results["test_case_4"] = low_rank_svd_approximation(A, k, full_matrices=False)

    return results

test_results = test_low_rank_svd_approximation()
