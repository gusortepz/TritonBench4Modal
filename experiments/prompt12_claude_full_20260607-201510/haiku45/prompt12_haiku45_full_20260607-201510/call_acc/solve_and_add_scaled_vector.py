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


def solve_and_add_scaled_vector(A: Tensor, b: Tensor, y: Tensor, alpha: float) -> Tensor:
    """
    Solves the triangular system Ax = b, then adds alpha * y to the solution.
    
    Args:
        A: Triangular matrix of shape (n, n)
        b: Right-hand side of shape (n,) or (n, k)
        y: Vector to be scaled and added, shape (n,) or broadcastable
        alpha: Scaling factor
        
    Returns:
        Solution x + alpha * y
    """
    # Solve the triangular system using PyTorch's built-in solver
    # torch.linalg.solve_triangular solves A @ x = b for upper triangular A
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add the scaled vector to the solution
    # y will be broadcast to match x's shape if needed
    result = x + alpha * y
    
    return result

##################################################################################################################################################



import torch

def test_solve_and_add_scaled_vector():
    results = {}

    # Test case 1: Basic test with 2x2 upper triangular matrix
    A1 = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device='cuda')
    b1 = torch.tensor([[5.0, 6.0], [7.0, 8]], device='cuda')
    y1 = torch.tensor([1.0, 2.0], device='cuda')
    alpha1 = 0.5
    results["test_case_1"] = solve_and_add_scaled_vector(A1, b1, y1, alpha1)
    return results

test_results = test_solve_and_add_scaled_vector()
