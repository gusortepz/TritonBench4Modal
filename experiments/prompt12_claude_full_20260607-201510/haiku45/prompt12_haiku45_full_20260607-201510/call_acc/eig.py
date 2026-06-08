import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple

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


def eig(A: Tensor, *, out: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the eigenvalue decomposition of a square matrix.
    
    Args:
        A: tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
           consisting of diagonalizable matrices.
        out: optional tuple of two tensors for output. If provided, results are copied into
             these tensors and the tuple is returned.
    
    Returns:
        A tuple (eigenvalues, eigenvectors) where:
        - eigenvalues: shape `(*, n)` containing the eigenvalues
        - eigenvectors: shape `(*, n, n)` containing the eigenvectors as columns
    """
    # Compute eigendecomposition using PyTorch's linalg.eig
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Handle out parameter
    if out is not None:
        out_eigenvalues, out_eigenvectors = out
        out_eigenvalues.copy_(eigenvalues)
        out_eigenvectors.copy_(eigenvectors)
        return out_eigenvalues, out_eigenvectors
    
    return eigenvalues, eigenvectors

##################################################################################################################################################



import torch

# def eig(A):
#     (eigenvalues, eigenvectors) = torch.linalg.eig(A)
#     return (eigenvalues, eigenvectors)

def test_eig():
    results = {}

    # Test case 1: 2x2 matrix with distinct eigenvalues
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    results["test_case_1"] = eig(A1)

    # Test case 2: 2x2 matrix with repeated eigenvalues
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = eig(A2)

    # Test case 3: 3x3 matrix with complex eigenvalues
    A3 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
    results["test_case_3"] = eig(A3)

    # Test case 4: 3x3 matrix with real eigenvalues
    A4 = torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]], device='cuda')
    results["test_case_4"] = eig(A4)

    return results

test_results = test_eig()
