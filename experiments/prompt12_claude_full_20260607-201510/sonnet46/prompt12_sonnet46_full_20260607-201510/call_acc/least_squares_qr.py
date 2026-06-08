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


def least_squares_qr(A, b, *, mode='reduced', out=None) -> Tensor:
    """
    Solves the least squares problem min |Ax - b|_2 using QR decomposition.
    
    Args:
        A: Coefficient matrix of shape (*, m, n)
        b: Right-hand side of shape (*, m) or (*, m, k)
        mode: QR mode, 'reduced' or 'complete'
        out: Optional output tensor
    
    Returns:
        Least squares solution x of shape (*, n) or (*, n, k)
    """
    # Determine if b is a vector or matrix
    b_is_vector = (b.dim() == A.dim() - 1)
    
    # If b is a vector, unsqueeze to make it a column matrix
    if b_is_vector:
        b_mat = b.unsqueeze(-1)
    else:
        b_mat = b
    
    # Compute QR decomposition of A
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute Q^T @ b
    # Q has shape (*, m, k) where k = min(m, n) for 'reduced' or m for 'complete'
    Qt_b = torch.matmul(Q.mT, b_mat)
    
    # Get the shape info
    # R has shape (*, k, n) where k = min(m, n) for 'reduced' or m for 'complete'
    # We only need the top-n rows of R and corresponding Qt_b entries
    n = A.shape[-1]
    
    # For 'complete' mode, R has shape (*, m, n) and Qt_b has shape (*, m, k)
    # We take the first n rows
    R_top = R[..., :n, :]  # (*, n, n)
    Qt_b_top = Qt_b[..., :n, :]  # (*, n, k)
    
    # Solve the upper triangular system R @ x = Q^T @ b
    # torch.linalg.solve_triangular expects (*, n, n) and (*, n, k)
    x = torch.linalg.solve_triangular(R_top, Qt_b_top, upper=True)
    
    # If b was a vector, squeeze the last dimension
    if b_is_vector:
        x = x.squeeze(-1)
    
    if out is not None:
        out.copy_(x)
        return out
    return x

##################################################################################################################################################



import torch

def test_least_squares_qr():
    results = {}
    
    # Test case 1: Simple overdetermined system with reduced QR
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b1 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    results["test_case_1"] = least_squares_qr(A1, b1)
    
    # Test case 4: Multiple right-hand sides with reduced QR
    A4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b4 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    results["test_case_4"] = least_squares_qr(A4, b4)
    
    return results

# Run the test
test_results = test_least_squares_qr()
