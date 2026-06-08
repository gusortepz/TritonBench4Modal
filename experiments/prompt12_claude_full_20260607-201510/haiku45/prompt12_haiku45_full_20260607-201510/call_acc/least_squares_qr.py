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


def least_squares_qr(A: Tensor, b: Tensor, *, mode: str = 'reduced', out: Optional[Tensor] = None) -> Tensor:
    """
    Solves the least squares problem for an overdetermined system using QR decomposition.
    
    Computes the least squares solution x that minimizes |Ax - b|_2.
    
    Args:
        A: Coefficient matrix of shape (*, m, n)
        b: Right-hand side of shape (*, m) or (*, m, k)
        mode: 'reduced' (default) or 'complete' for QR decomposition
        out: Optional output tensor
        
    Returns:
        Solution x of shape (*, n) or (*, n, k) depending on b's shape
    """
    # Input validation
    if A.dim() < 2:
        raise ValueError(f"A must have at least 2 dimensions, got {A.dim()}")
    if b.dim() < 1:
        raise ValueError(f"b must have at least 1 dimension, got {b.dim()}")
    
    # Ensure b has compatible batch dimensions and m dimension
    batch_shape_A = A.shape[:-2]
    batch_shape_b = b.shape[:-1] if b.dim() > 1 else ()
    
    m, n = A.shape[-2:]
    
    if b.dim() == 1:
        # b is (m,), reshape to (m, 1) for computation
        b_original_shape = b.shape
        b = b.unsqueeze(-1)
    else:
        b_original_shape = None
    
    # Validate shapes match
    if A.shape[-2] != b.shape[-2]:
        raise ValueError(f"A and b must have matching first dimension on last axis: {A.shape[-2]} vs {b.shape[-2]}")
    
    # Compute QR decomposition
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Q shape: (*, m, m) if mode='complete', (*, m, n) if mode='reduced'
    # R shape: (*, m, n) if mode='complete', (*, n, n) if mode='reduced'
    
    # Compute Q^T @ b
    # Q: (*, m, q) where q = m (complete) or n (reduced)
    # b: (*, m, k)
    # Q^T @ b: (*, q, k)
    QtB = torch.matmul(Q.transpose(-2, -1), b)
    
    # Solve R @ x = Q^T @ b using triangular solve
    # R: (*, m, n) or (*, n, n) depending on mode
    # QtB: (*, m, k) or (*, n, k)
    
    if mode == 'reduced':
        # R is (*, n, n), QtB is (*, n, k)
        # Solve R @ x = QtB for x
        x = torch.linalg.solve_triangular(R, QtB, upper=True)
    else:
        # R is (*, m, n), QtB is (*, m, k)
        # We only need the first n rows of R and QtB
        R_reduced = R[..., :n, :n]
        QtB_reduced = QtB[..., :n, :]
        x = torch.linalg.solve_triangular(R_reduced, QtB_reduced, upper=True)
    
    # Reshape output if b was originally 1D
    if b_original_shape is not None:
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
