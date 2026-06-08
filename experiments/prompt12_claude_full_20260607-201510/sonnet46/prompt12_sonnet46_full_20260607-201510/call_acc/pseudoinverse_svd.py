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


def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> Tensor:
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.
    
    Args:
        A: Input tensor of shape (*, m, n)
        full_matrices: If True, compute full SVD. If False, compute reduced SVD.
        rcond: Relative condition number threshold.
        out: Optional output tensor.
    
    Returns:
        Pseudoinverse of A with shape (*, n, m)
    """
    # Use torch.linalg.svd to decompose A
    # full_matrices=False gives reduced SVD which is more efficient for pseudoinverse
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Compute threshold for singular values
    # rcond * max_singular_value
    # S has shape (*, k) where k = min(m, n)
    # We need to handle batch dimensions
    if S.numel() == 0:
        # Edge case: empty singular values
        result = torch.zeros(
            A.shape[:-2] + (A.shape[-1], A.shape[-2]),
            dtype=A.dtype,
            device=A.device
        )
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # Get the maximum singular value along the last dimension
    # S shape: (*, k)
    max_sv = S.amax(dim=-1, keepdim=True)  # (*, 1)
    
    # Threshold
    threshold = rcond * max_sv  # (*, 1)
    
    # Invert singular values above threshold, zero out the rest
    # Handle complex dtypes: S is always real even for complex A
    S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
    
    # Reconstruct pseudoinverse: A^+ = Vh^H @ diag(S_inv) @ U^H
    # Vh shape: (*, k, n) if full_matrices=False, or (*, n, n) if full_matrices=True
    # U shape: (*, m, k) if full_matrices=False, or (*, m, m) if full_matrices=True
    # We need: A^+ = Vh^H @ diag(S_inv) @ U^H
    
    # For full_matrices=True case, we need to handle the extra rows/columns
    # The pseudoinverse only uses the first k columns/rows
    m = A.shape[-2]
    n = A.shape[-1]
    k = S.shape[-1]  # min(m, n) for reduced, or max(m, n) for full... 
    # Actually k = min(m, n) for S regardless of full_matrices
    
    # U: (*, m, m) if full_matrices else (*, m, k)
    # Vh: (*, n, n) if full_matrices else (*, k, n)
    # S: (*, k) where k = min(m, n)
    
    # Take only first k columns of U and first k rows of Vh
    U_k = U[..., :k]    # (*, m, k)
    Vh_k = Vh[..., :k, :]  # (*, k, n)
    
    # Compute: result = Vh_k^H @ diag(S_inv) @ U_k^H
    # = Vh_k^conj_transpose @ diag(S_inv) @ U_k^conj_transpose
    
    # S_inv: (*, k) -> broadcast as (*, k, 1) for scaling
    # Scale rows of Vh_k^H by S_inv:
    # Vh_k^H shape: (*, n, k)
    # diag(S_inv) @ U_k^H: scale rows of U_k^H by S_inv
    # U_k^H shape: (*, k, m)
    
    # Efficient computation:
    # A^+ = (Vh_k^H) @ diag(S_inv) @ (U_k^H)
    # = (Vh_k.conj().transpose(-2,-1)) @ diag(S_inv) @ (U_k.conj().transpose(-2,-1))
    
    Vh_k_H = Vh_k.conj().transpose(-2, -1)  # (*, n, k)
    U_k_H = U_k.conj().transpose(-2, -1)    # (*, k, m)
    
    # Scale: Vh_k_H * S_inv[..., None, :] gives (*, n, k) scaled by S_inv
    # Then matmul with U_k_H (*, k, m)
    S_inv_expanded = S_inv.unsqueeze(-2)  # (*, 1, k)
    
    # (*, n, k) * (*, 1, k) = (*, n, k)
    scaled_Vh_H = Vh_k_H * S_inv_expanded
    
    # (*, n, k) @ (*, k, m) = (*, n, m)
    result = torch.matmul(scaled_Vh_H, U_k_H)
    
    if out is not None:
        out.copy_(result)
        return out
    return result

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
