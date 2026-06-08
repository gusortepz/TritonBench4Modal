import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Union
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


def matrix_power_eig(A: Tensor, k: Union[float, complex], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the matrix power A^k using eigendecomposition.
    
    A^k = V diag(Λ^k) V^(-1)
    where Λ and V are eigenvalues and eigenvectors of A.
    
    Args:
        A (Tensor): tensor of shape `(*, n, n)` of square matrices
        k (float or complex): exponent
        out (Tensor, optional): output tensor
    
    Returns:
        Tensor: A^k with same shape as A
    """
    # Compute eigendecomposition
    # torch.linalg.eig returns (eigenvalues, eigenvectors)
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Compute eigenvalues^k
    # Handle both real and complex k
    if isinstance(k, complex):
        # Convert to complex if k is complex
        eigenvalues_powered = eigenvalues ** k
    else:
        # For real k, keep dtype if possible
        eigenvalues_powered = eigenvalues ** k
    
    # Compute V @ diag(Λ^k) @ V^(-1)
    # diag(Λ^k) @ V^(-1) can be computed as scaling rows of V^(-1)
    V_inv = torch.linalg.inv(eigenvectors)
    
    # Scale V_inv rows by eigenvalues_powered
    # eigenvalues_powered shape: (*, n) or batch + (n,)
    # V_inv shape: (*, n, n)
    # We need to multiply each row of V_inv by the corresponding eigenvalue power
    scaled_V_inv = eigenvalues_powered.unsqueeze(-1) * V_inv
    
    # Compute V @ (diag(Λ^k) @ V^(-1))
    result = torch.matmul(eigenvectors, scaled_V_inv)
    
    # Handle output tensor
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
