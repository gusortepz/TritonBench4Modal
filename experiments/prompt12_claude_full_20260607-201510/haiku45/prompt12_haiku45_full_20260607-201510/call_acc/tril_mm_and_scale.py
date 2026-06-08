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


def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Performs matrix multiplication of the lower triangular part of A with B,
    scales by alpha, then scales the result by beta.
    
    Args:
        A: A 2D matrix of shape (n, n)
        B: A matrix of shape (n, p)
        alpha: Scaling factor for the matrix multiplication result
        beta: Scaling factor for the final result
    
    Returns:
        Result tensor of shape (n, p)
    """
    # Extract lower triangular part of A
    A_tril = torch.tril(A)
    
    # Perform matrix multiplication: A_tril @ B
    result = torch.matmul(A_tril, B)
    
    # Scale by alpha
    result = result * alpha
    
    # Scale by beta
    result = result * beta
    
    return result

##################################################################################################################################################



import torch

def test_tril_mm_and_scale():
    results = {}

    # Test case 1: Basic functionality with square matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = tril_mm_and_scale(A1, B1, alpha1, beta1)

    # Test case 2: Different alpha and beta values
    A2 = torch.tensor([[1.0, 0.0], [3.0, 4.0]], device='cuda')
    B2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = tril_mm_and_scale(A2, B2, alpha2, beta2)

    # Test case 3: Larger matrix
    A3 = torch.tensor([[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]], device='cuda')
    B3 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = tril_mm_and_scale(A3, B3, alpha3, beta3)

    # Test case 4: Zero matrix A
    A4 = torch.zeros((2, 2), device='cuda')
    B4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = tril_mm_and_scale(A4, B4, alpha4, beta4)

    return results

test_results = test_tril_mm_and_scale()
