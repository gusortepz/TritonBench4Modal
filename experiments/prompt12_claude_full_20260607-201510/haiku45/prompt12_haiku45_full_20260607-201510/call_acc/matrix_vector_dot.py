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


def matrix_vector_dot(A: Tensor, x: Tensor, y: Tensor, alpha: float, beta: float) -> Tensor:
    """
    Computes the matrix-vector product y = alpha * torch.mv(A, x) + beta * y,
    then returns the dot product torch.dot(y, x).
    
    Args:
        A (Tensor): The input matrix of shape (n, m).
        x (Tensor): The input vector of shape (m,).
        y (Tensor): The target vector to be modified, of shape (n,).
        alpha (float): Scalar multiplier for torch.mv(A, x).
        beta (float): Scalar multiplier for y.
    
    Returns:
        Tensor: The dot product of the updated y with x.
    """
    # Compute matrix-vector product: A @ x
    mv_result = torch.mv(A, x)
    
    # Update y in-place: y = alpha * (A @ x) + beta * y
    y.mul_(beta).add_(mv_result, alpha=alpha)
    
    # Compute and return dot product: y · x
    return torch.dot(y, x)

##################################################################################################################################################



import torch
from torch import Tensor

def test_matrix_vector_dot():
    results = {}
    
    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([0.0, 0.0], device='cuda')
    alpha = 1.0
    beta = 0.0
    results["test_case_1"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_2"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 3
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([2.0, 3.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 0.5
    beta = 0.5
    results["test_case_3"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 4
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([2.0, 2.0], device='cuda')
    alpha = 2.0
    beta = 0.5
    results["test_case_4"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    return results

test_results = test_matrix_vector_dot()
