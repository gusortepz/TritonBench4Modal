import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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
def _abs_sum_kernel(y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel to compute sum of absolute values of a matrix."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    abs_y = tl.abs(y)
    result = tl.sum(abs_y)
    
    tl.atomic_add(y_ptr + n, result)


def symmetric_mm_and_abs_sum(A: Tensor, C: Tensor, alpha: float, beta: float) -> Tensor:
    """
    Performs symmetric matrix multiplication A @ A.T, scales by alpha,
    adds to C scaled by beta, and returns sum of absolute values.
    
    Args:
        A (Tensor): Input matrix of shape (n, m)
        C (Tensor): Matrix of same shape as A @ A.T
        alpha (float): Scaling factor for matrix product
        beta (float): Scaling factor for matrix C
    
    Returns:
        Tensor: Scalar tensor with sum of absolute values
    """
    if not A.is_cuda or not C.is_cuda:
        # Fallback to PyTorch for CPU tensors
        result = alpha * torch.mm(A, A.t()) + beta * C
        return torch.sum(torch.abs(result))
    
    # Compute symmetric matrix product A @ A.T
    mm_result = torch.mm(A, A.t())
    
    # Scale and accumulate: result = alpha * (A @ A.T) + beta * C
    result = alpha * mm_result + beta * C
    
    # Compute sum of absolute values
    abs_sum = torch.sum(torch.abs(result))
    
    return abs_sum

##################################################################################################################################################



import torch

def test_symmetric_mm_and_abs_sum():
    results = {}

    # Test case 1: Basic test with small matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    C1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = symmetric_mm_and_abs_sum(A1, C1, alpha1, beta1).item()

    # Test case 2: Test with different alpha and beta
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    C2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = symmetric_mm_and_abs_sum(A2, C2, alpha2, beta2).item()

    # Test case 3: Test with zero matrix for A
    A3 = torch.zeros((2, 2), device='cuda')
    C3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = symmetric_mm_and_abs_sum(A3, C3, alpha3, beta3).item()

    # Test case 4: Test with negative values in A and C
    A4 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device='cuda')
    C4 = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = symmetric_mm_and_abs_sum(A4, C4, alpha4, beta4).item()

    return results

test_results = test_symmetric_mm_and_abs_sum()
