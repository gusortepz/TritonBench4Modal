import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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


def chebyshev_polynomial_t(input: Tensor, n: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the Chebyshev polynomial of the first kind T_n(input).
    
    For n = 0, returns 1.
    For n = 1, returns input.
    For n < 6 or |input| > 1, uses recursive formula: T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    Otherwise, uses trigonometric formula: T_n(x) = cos(n * arccos(x))
    
    Args:
        input: the input tensor
        n: degree of the polynomial (scalar or tensor)
        out: optional output tensor
    
    Returns:
        Tensor of Chebyshev polynomial values
    """
    
    # Ensure n is a scalar integer
    if isinstance(n, Tensor):
        if n.numel() != 1:
            raise ValueError("n must be a scalar")
        n_val = n.item()
    else:
        n_val = int(n)
    
    # Base cases
    if n_val == 0:
        y = torch.ones_like(input)
    elif n_val == 1:
        y = input.clone()
    elif n_val < 6 or (input.abs() > 1).any():
        # Use recursive formula for small n or when |input| > 1
        y = _chebyshev_recursive(input, n_val)
    else:
        # Use trigonometric formula for |input| <= 1 and n >= 6
        y = _chebyshev_trig(input, n_val)
    
    if out is not None:
        out.copy_(y)
        return out
    return y


def _chebyshev_recursive(input: Tensor, n: int) -> Tensor:
    """
    Compute Chebyshev polynomial using recursive formula.
    T_{k+1}(x) = 2*x*T_k(x) - T_{k-1}(x)
    """
    if n == 0:
        return torch.ones_like(input)
    elif n == 1:
        return input.clone()
    
    t_prev2 = torch.ones_like(input)
    t_prev1 = input.clone()
    
    for k in range(1, n):
        t_curr = 2.0 * input * t_prev1 - t_prev2
        t_prev2 = t_prev1
        t_prev1 = t_curr
    
    return t_prev1


def _chebyshev_trig(input: Tensor, n: int) -> Tensor:
    """
    Compute Chebyshev polynomial using trigonometric formula.
    T_n(x) = cos(n * arccos(x))
    Valid for |x| <= 1.
    """
    # Clamp input to [-1, 1] for numerical stability in arccos
    x_clamped = torch.clamp(input, -1.0, 1.0)
    theta = torch.acos(x_clamped)
    return torch.cos(n * theta)

##################################################################################################################################################



import torch

def test_chebyshev_polynomial_t():
    results = {}

    # Test case 1: Basic test with n=0
    input_tensor_1 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_1 = 0
    results["test_case_1"] = chebyshev_polynomial_t(input_tensor_1, n_1)

    # Test case 2: Basic test with n=1
    input_tensor_2 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_2 = 1
    results["test_case_2"] = chebyshev_polynomial_t(input_tensor_2, n_2)

    # Test case 3: Higher degree polynomial n=3
    input_tensor_3 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_3 = 3
    results["test_case_3"] = chebyshev_polynomial_t(input_tensor_3, n_3)

    # Test case 4: Negative input values with n=2
    input_tensor_4 = torch.tensor([-1.0, -0.5, -0.2], device='cuda')
    n_4 = 2
    results["test_case_4"] = chebyshev_polynomial_t(input_tensor_4, n_4)

    return results

test_results = test_chebyshev_polynomial_t()
