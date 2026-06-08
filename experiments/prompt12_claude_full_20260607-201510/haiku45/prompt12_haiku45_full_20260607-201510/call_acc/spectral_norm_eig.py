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


def spectral_norm_eig(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the spectral norm (operator norm) of a square matrix using eigenvalues.
    
    The spectral norm is the largest absolute value among the eigenvalues of the matrix.
    
    Args:
        A (Tensor): Tensor of shape (*, n, n) where * is zero or more batch dimensions.
                   Supported dtypes: float32, float64, complex64, complex128.
        out (Tensor, optional): Output tensor. If provided, the result is written to it.
                               Default: None.
    
    Returns:
        Tensor: Spectral norm of shape (*,) containing the largest absolute eigenvalue
               for each matrix in the batch.
    """
    if A.dim() < 2:
        raise ValueError(f"Input must be at least 2-D, got {A.dim()}-D")
    
    if A.shape[-2] != A.shape[-1]:
        raise ValueError(f"Input must be square, got shape {A.shape}")
    
    # Compute eigenvalues using torch.linalg.eigvalsh for symmetric/Hermitian
    # or torch.linalg.eigvals for general matrices
    if A.dtype in (torch.complex64, torch.complex128):
        # For complex matrices, use eigvals (general eigendecomposition)
        eigenvalues = torch.linalg.eigvals(A)
    else:
        # For real matrices, treat as Hermitian (or use eigvals for asymmetric)
        # We use eigvals for correctness in case the matrix is not symmetric
        eigenvalues = torch.linalg.eigvals(A)
    
    # Compute absolute values of eigenvalues
    abs_eigenvalues = torch.abs(eigenvalues)
    
    # Get the maximum absolute eigenvalue for each matrix in the batch
    # Result shape: (*)
    y = torch.amax(abs_eigenvalues, dim=-1)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_spectral_norm_eig():
    results = {}

    # Test case 1: Single 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = spectral_norm_eig(A1)

    # Test case 2: Batch of 2x2 matrices
    A2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_2"] = spectral_norm_eig(A2)

    # Test case 3: Single 3x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_3"] = spectral_norm_eig(A3)

    # Test case 4: Batch of 3x3 matrices
    A4 = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 
                       [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]], device='cuda')
    results["test_case_4"] = spectral_norm_eig(A4)

    return results

test_results = test_spectral_norm_eig()
