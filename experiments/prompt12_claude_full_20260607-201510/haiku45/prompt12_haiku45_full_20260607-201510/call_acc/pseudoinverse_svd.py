import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl
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


def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> Tensor:
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.
    
    Args:
        A (Tensor): Input tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    
    Keyword args:
        full_matrices (bool, optional): If `True` (default), compute the full SVD. 
                                       If `False`, compute the reduced SVD.
        rcond (float, optional): Relative condition number threshold. Singular values 
                                smaller than `rcond * largest_singular_value` are set to zero. 
                                Default: `1e-15`.
        out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
    
    Returns:
        Tensor: The pseudoinverse of A with shape `(*, n, m)`.
    """
    
    # Use torch.linalg.svd for the decomposition
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Compute threshold for singular values
    # rcond * max_singular_value
    if S.numel() > 0:
        max_s = torch.amax(S, dim=-1, keepdim=True)
        threshold = rcond * max_s
    else:
        threshold = torch.tensor(rcond, dtype=S.dtype, device=S.device)
    
    # Invert non-zero singular values: 1/s if s > threshold, else 0
    S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
    
    # Reconstruct pseudoinverse: V @ diag(S_inv) @ U^H
    # Vh is already V^H, so we need V = Vh^H
    # pseudoinverse = Vh^H @ diag(S_inv) @ U^H
    
    # For complex tensors, use .conj().transpose(-2, -1) for conjugate transpose
    # For real tensors, this is just transpose
    if A.is_complex():
        U_conj_t = U.conj().transpose(-2, -1)
        Vh_conj_t = Vh.conj().transpose(-2, -1)
    else:
        U_conj_t = U.transpose(-2, -1)
        Vh_conj_t = Vh.transpose(-2, -1)
    
    # Build the diagonal matrix of S_inv: reshape for batch matmul
    # S_inv has shape (..., min(m, n))
    # We need to multiply: Vh_conj_t @ diag(S_inv) @ U_conj_t
    # Which is: Vh_conj_t @ (S_inv * U_conj_t)
    
    # Expand S_inv to match U_conj_t for broadcasting
    S_inv_expanded = S_inv.unsqueeze(-2)  # (..., 1, min(m,n))
    
    # Element-wise multiply and then matmul
    temp = S_inv_expanded * U_conj_t  # (..., 1, min(m,n)) * (..., min(m,n), m) -> (..., 1, m) after broadcast
    # Actually: (..., 1, min(m,n)) * (..., min(m,n), m) requires careful broadcasting
    
    # Better approach: use diagonal matrix multiplication
    # pseudoinv = Vh^H @ diag(S_inv) @ U^H
    # = (U @ diag(S_inv) @ Vh)^H
    # But we have U, S, Vh from torch.linalg.svd
    
    # Correct reconstruction:
    # A^+ = V @ Sigma_inv @ U^H
    # Given: U, S, Vh where A = U @ diag(S) @ Vh
    # Then: A^+ = Vh^H @ diag(S_inv) @ U^H
    
    # Create diagonal scaling: multiply Vh by S_inv along the appropriate dimension
    # Vh shape: (..., min(m,n), n) or (..., m, n) depending on full_matrices
    # S_inv shape: (..., min(m,n))
    
    # Method: Vh_conj_t @ diag(S_inv) @ U_conj_t
    # = (S_inv * U_conj_t)^T @ Vh_conj_t (after transpose)
    # Simpler: (U @ diag(S_inv)) @ Vh, then conjugate transpose
    
    k = S_inv.shape[-1]  # min(m, n)
    m = U.shape[-2]
    n = Vh.shape[-1]
    
    # U: (..., m, m) or (..., m, k)
    # S_inv: (..., k)
    # Vh: (..., k, n) or (..., m, n)
    
    # Multiply U by diagonal S_inv: U @ diag(S_inv)
    U_scaled = U[..., :k, :] * S_inv.unsqueeze(-2)  # (..., k, m) -> (..., k, m) with scaling
    
    # Then multiply by Vh: (U @ diag(S_inv)) @ Vh^H
    # (U_scaled)^H @ Vh^H = (Vh @ U_scaled)^H
    # Or: U_scaled^H @ Vh^H
    
    # Simpler direct computation using torch.linalg.svd semantics:
    # A^+ = Vh^H @ diag(S_inv) @ U^H
    
    # Compute diag(S_inv) @ U^H first, then Vh^H @ result
    # diag(S_inv) @ U^H: scale rows of U^H
    U_t = U_conj_t  # (..., m, m) or (..., m, k)
    S_inv_expanded = S_inv.unsqueeze(-2)  # (..., 1, k)
    
    # For full_matrices=True: U is (..., m, m), U_t is (..., m, m)
    # Need to select only first k rows: U_t[..., :k, :] has shape (..., k, m)
    temp = S_inv_expanded * U_t[..., :k, :]  # (..., 1, k) * (..., k, m) -> (..., k, m)
    
    # Now Vh_conj_t @ temp = (..., n, k) @ (..., k, m) -> (..., n, m)
    pinv = Vh_conj_t @ temp
    
    if out is not None:
        out.copy_(pinv)
        return out
    return pinv

##################################################################################################################################################



import torch

def test_pseudoinverse_svd():
    results = {}

    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = pseudoinverse_svd(A1)

    # Test case 4: Singular matrix
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    results["test_case_4"] = pseudoinverse_svd(A4)

    return results

test_results = test_pseudoinverse_svd()
