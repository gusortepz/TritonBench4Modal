import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

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


def symmetric_matrix_vector_norm(
    A: torch.Tensor,
    x: torch.Tensor,
    alpha: float,
    beta: float,
    p: float = 2.0,
) -> torch.Tensor:
    """
    Computes the matrix-vector product for a symmetric matrix A and vector x,
    with scaling factors alpha and beta, then calculates the norm of the result.
    
    Operation:
    1. y = alpha * torch.mv(A, x) + beta * y
    2. norm = torch.norm(y, p)
    
    Args:
        A: Symmetric matrix of shape (n, n)
        x: Vector of shape (n,)
        alpha: Scalar multiplier for the matrix-vector product
        beta: Scalar multiplier for the initial y (assumed zero if not provided)
        p: Order of the norm (default 2.0 for Euclidean norm)
    
    Returns:
        Scalar tensor containing the norm of the result vector
    """
    # Compute matrix-vector product: y = alpha * A @ x + beta * y
    # Since y is not provided as input, we assume initial y = 0
    # Therefore: y = alpha * A @ x
    y = torch.mv(A, x)
    y = alpha * y
    
    # Compute the norm of y
    norm = torch.norm(y, p=p)
    
    return norm

##################################################################################################################################################



import torch

def test_symmetric_matrix_vector_norm():
    results = {}

    # Test case 1: Basic test with default p value
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_1"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 2: Different alpha and beta values
    alpha = 2.0
    beta = 0.5
    results["test_case_2"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 3: Different p value (1-norm)
    alpha = 1.0
    beta = 1.0
    p = 1.0
    results["test_case_3"] = symmetric_matrix_vector_norm(A, x, alpha, beta, p).item()

    # Test case 4: Larger matrix and vector
    A = torch.tensor([[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 3.0]], device='cuda')
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha = 1.5
    beta = 0.5
    results["test_case_4"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    return results

test_results = test_symmetric_matrix_vector_norm()
