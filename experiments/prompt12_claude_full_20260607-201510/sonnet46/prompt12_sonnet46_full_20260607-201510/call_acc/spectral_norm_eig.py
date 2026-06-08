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


def spectral_norm_eig(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the spectral norm (largest absolute eigenvalue) of a square matrix.
    
    Supports batches of matrices of shape (*, n, n).
    Supports float, double, cfloat, cdouble dtypes.
    
    Args:
        A: Tensor of shape (*, n, n)
        out: Optional output tensor
    
    Returns:
        Tensor of shape (*,) containing the spectral norm of each matrix.
    """
    # Validate input is at least 2D and square
    if A.dim() < 2:
        raise ValueError(f"Expected A to be at least 2D, got {A.dim()}D tensor")
    
    n, m = A.shape[-2], A.shape[-1]
    if n != m:
        raise ValueError(f"Expected square matrices, got shape (..., {n}, {m})")
    
    # Compute eigenvalues; eigvals supports complex and real inputs
    # For real inputs, eigenvalues may be complex
    # torch.linalg.eigvals returns complex eigenvalues
    eigenvalues = torch.linalg.eigvals(A)
    
    # Spectral norm = max absolute eigenvalue
    # abs of complex eigenvalue is the modulus
    abs_eigenvalues = torch.abs(eigenvalues)
    
    # Take max over last dimension (eigenvalue dimension)
    result = abs_eigenvalues.max(dim=-1).values
    
    # For real input dtypes, the result should be real
    # abs_eigenvalues is always real, so result is real
    # Convert to appropriate real dtype based on input
    if A.dtype == torch.float32 or A.dtype == torch.complex64:
        result = result.to(torch.float32)
    elif A.dtype == torch.float64 or A.dtype == torch.complex128:
        result = result.to(torch.float64)
    
    if out is not None:
        out.copy_(result)
        return out
    return result

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
