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
def _row_dot_kernel(
    row1_ptr,
    row2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute dot product of two rows using Triton."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row1 = tl.load(row1_ptr + offsets, mask=mask, other=0.0)
    row2 = tl.load(row2_ptr + offsets, mask=mask, other=0.0)

    product = row1 * row2
    sum_val = tl.sum(product)

    tl.atomic_add(out_ptr, sum_val)


def _matrix_multiply_and_row_dot_impl(
    A: Tensor, B: Tensor, alpha: float, beta: float, C: Tensor
) -> Tensor:
    """Reference implementation: scaled matmul + add scaled C + row dot product."""
    # Compute alpha * A @ B + beta * C
    result = alpha * torch.matmul(A, B) + beta * C

    # Compute dot product of first two rows
    if result.shape[0] < 2:
        # If fewer than 2 rows, return a zero tensor
        return torch.tensor(0.0, dtype=result.dtype, device=result.device)

    row_dot = torch.dot(result[0, :], result[1, :])
    return row_dot


try:
    _matrix_multiply_and_row_dot_fast = torch.compile(
        _matrix_multiply_and_row_dot_impl, mode="max-autotune", fullgraph=False
    )
except Exception:
    _matrix_multiply_and_row_dot_fast = _matrix_multiply_and_row_dot_impl


def matrix_multiply_and_row_dot(
    A: Tensor, B: Tensor, alpha: float, beta: float, C: Tensor
) -> Tensor:
    """
    Computes a scaled matrix-matrix product, then calculates the dot product
    of the first two rows of the resulting matrix.

    Args:
        A (Tensor): First input matrix of shape (n, m).
        B (Tensor): Second input matrix of shape (m, p).
        alpha (float): Scalar multiplier for the matrix-matrix product.
        beta (float): Scalar multiplier for the input matrix C.
        C (Tensor): Matrix of shape (n, p) where the results are added.

    Returns:
        Tensor: A scalar tensor containing the dot product of the first two rows
                of the result (alpha * A @ B + beta * C).
    """
    try:
        return _matrix_multiply_and_row_dot_fast(A, B, alpha, beta, C)
    except Exception:
        return _matrix_multiply_and_row_dot_impl(A, B, alpha, beta, C)

##################################################################################################################################################



import torch

def test_matrix_multiply_and_row_dot():
    results = {}

    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 1.0
    beta = 0.0
    C = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device='cuda')
    results["test_case_1"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 0.5
    beta = 0.5
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_2"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 3
    A = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    B = torch.tensor([[6.0, 7.0], [8.0, 9.0]], device='cuda')
    alpha = 1.0
    beta = 1.0
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_3"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 4
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha = 2.0
    beta = 0.5
    C = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device='cuda')
    results["test_case_4"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    return results

test_results = test_matrix_multiply_and_row_dot()
