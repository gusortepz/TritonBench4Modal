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


@triton.jit
def _scale_add_kernel(
    mv_ptr,
    y_ptr,
    out_ptr,
    alpha,
    beta,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    mv_val = tl.load(mv_ptr + offsets, mask=mask)
    y_val = tl.load(y_ptr + offsets, mask=mask)
    result = alpha * mv_val + beta * y_val
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _dot_kernel(
    y_ptr,
    x_ptr,
    out_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    prod = y_val * x_val
    # Use atomic add to accumulate into out
    tl.atomic_add(out_ptr, tl.sum(prod, axis=0))


def _matrix_vector_dot_impl(A: Tensor, x: Tensor, y: Tensor, alpha: float, beta: float) -> Tensor:
    # Compute mv(A, x)
    mv_result = torch.mv(A, x)
    # Update y in-place: y = alpha * mv(A, x) + beta * y
    y.mul_(beta).add_(mv_result, alpha=alpha)
    # Return dot(y, x)
    return torch.dot(y, x)


def matrix_vector_dot(A: Tensor, x: Tensor, y: Tensor, alpha: float, beta: float) -> Tensor:
    """
    Computes y = alpha * mv(A, x) + beta * y in-place, then returns dot(y, x).

    Args:
        A (Tensor): The input matrix of shape (n, m).
        x (Tensor): The input vector of shape (m,).
        y (Tensor): The target vector to be modified, of shape (n,).
        alpha (float): Scalar multiplier for torch.mv(A, x).
        beta (float): Scalar multiplier for y.

    Returns:
        Tensor: The dot product of the updated y with x (scalar tensor).
    """
    if not A.is_cuda or not x.is_cuda or not y.is_cuda:
        # CPU path: use direct PyTorch
        mv_result = torch.mv(A, x)
        y.mul_(beta).add_(mv_result, alpha=alpha)
        return torch.dot(y, x)

    # CUDA path: use Triton for the scale-add step if n is reasonable
    n = y.shape[0]
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)

    try:
        # Compute mv(A, x) using cuBLAS
        mv_result = torch.mv(A, x)

        # Use Triton for the elementwise scale-add
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _scale_add_kernel[grid](
            mv_result,
            y,
            y,
            alpha,
            beta,
            n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Compute dot product using PyTorch (cuBLAS)
        result = torch.dot(y, x)
        return result
    except Exception:
        # Fallback to pure PyTorch
        return _matrix_vector_dot_impl(A, x, y, alpha, beta)

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
