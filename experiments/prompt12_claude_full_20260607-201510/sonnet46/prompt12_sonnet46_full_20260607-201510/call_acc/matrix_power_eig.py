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


def matrix_power_eig(A, k, *, out=None) -> Tensor:
    """
    Computes the matrix power A^k of a square matrix A using eigendecomposition.
    A^k = V diag(Λ^k) V^(-1)
    Supports fractional and complex exponents.
    """
    # Ensure we work with complex tensors for generality (especially fractional/complex k)
    if isinstance(k, complex) or (isinstance(k, float) and k != int(k)):
        # Use complex arithmetic
        A_complex = A.to(torch.complex128) if A.dtype not in (torch.complex64, torch.complex128) else A.to(torch.complex128)
    else:
        # Integer power or float that is an integer value
        if A.dtype in (torch.complex64, torch.complex128):
            A_complex = A.to(torch.complex128)
        else:
            # Try real first, fall back to complex if needed
            A_complex = A.to(torch.float64)

    # Compute eigendecomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eig(A_complex)
    except Exception:
        # Fallback: try with complex128
        A_complex = A.to(torch.complex128)
        eigenvalues, eigenvectors = torch.linalg.eig(A_complex)

    # Ensure complex for eigenvalue power computation
    if eigenvalues.dtype not in (torch.complex64, torch.complex128):
        eigenvalues = eigenvalues.to(torch.complex128)
    if eigenvectors.dtype not in (torch.complex64, torch.complex128):
        eigenvectors = eigenvectors.to(torch.complex128)

    # Compute Λ^k
    if isinstance(k, complex):
        k_tensor = torch.tensor(k, dtype=torch.complex128)
        eigenvalues_k = eigenvalues ** k_tensor
    else:
        # k is float or int
        k_val = float(k)
        # Use complex power for safety
        eigenvalues_k = eigenvalues ** k_val

    # Build diagonal matrix from eigenvalues^k
    # eigenvalues shape: (*, n)
    # We need diag matrix: (*, n, n)
    diag_k = torch.diag_embed(eigenvalues_k)  # (*, n, n)

    # Compute V @ diag(Λ^k) @ V^{-1}
    V_inv = torch.linalg.inv(eigenvectors)
    result_complex = eigenvectors @ diag_k @ V_inv

    # If original matrix was real and k is real, try to return real result
    if A.dtype in (torch.float32, torch.float64) and not isinstance(k, complex):
        # Check if imaginary part is negligible
        imag_max = result_complex.imag.abs().max().item()
        real_max = result_complex.real.abs().max().item()
        if imag_max < 1e-6 * (real_max + 1e-10):
            result = result_complex.real
            # Cast back to original dtype
            result = result.to(A.dtype)
        else:
            # Return complex result
            if A.dtype == torch.float32:
                result = result_complex.to(torch.complex64)
            else:
                result = result_complex.to(torch.complex128)
    elif A.dtype in (torch.complex64,):
        result = result_complex.to(torch.complex64)
    else:
        result = result_complex

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_matrix_power_eig():
    results = {}

    # Test case 1: Simple 2x2 matrix with integer exponent
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    k1 = 2
    results["test_case_1"] = matrix_power_eig(A1, k1)

    # Test case 2: 3x3 matrix with fractional exponent
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    k2 = 0.5
    results["test_case_2"] = matrix_power_eig(A2, k2)

    # Test case 4: Batch of 2x2 matrices with integer exponent
    A4 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    k4 = 3
    results["test_case_4"] = matrix_power_eig(A4, k4)

    return results

test_results = test_matrix_power_eig()
