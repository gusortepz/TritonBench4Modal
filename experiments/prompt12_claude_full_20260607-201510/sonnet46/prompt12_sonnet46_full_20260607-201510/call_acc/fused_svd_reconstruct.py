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


def fused_svd_reconstruct(A: Tensor) -> Tensor:
    """
    Reconstructs the input matrix A using its Singular Value Decomposition (SVD).
    
    1. Compute SVD: A = U @ diag(S) @ Vh
    2. Reconstruct: A_reconstructed = U @ diag(S) @ Vh
    
    Args:
        A: Input matrix of shape (m, n)
    
    Returns:
        Reconstructed matrix of shape (m, n)
    """
    # Compute the full SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Reconstruct A = U @ diag(S) @ Vh
    # U shape: (m, k), S shape: (k,), Vh shape: (k, n) where k = min(m, n)
    # Scale columns of U by singular values, then multiply by Vh
    A_reconstructed = (U * S.unsqueeze(-2)) @ Vh
    
    return A_reconstructed

##################################################################################################################################################



import torch

def test_fused_svd_reconstruct():
    results = {}
    
    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = fused_svd_reconstruct(A1)
    
    # Test case 2: Rectangular matrix (more rows than columns)
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_2"] = fused_svd_reconstruct(A2)
    
    # Test case 3: Rectangular matrix (more columns than rows)
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = fused_svd_reconstruct(A3)
    
    # Test case 4: Single value matrix
    A4 = torch.tensor([[42.0]], device='cuda')
    results["test_case_4"] = fused_svd_reconstruct(A4)
    
    return results

test_results = test_fused_svd_reconstruct()
