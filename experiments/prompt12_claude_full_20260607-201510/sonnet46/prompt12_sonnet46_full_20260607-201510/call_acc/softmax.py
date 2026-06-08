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
def _softmax_kernel_2d(
    input_ptr,
    output_ptr,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * row_stride
    out_start = output_ptr + row_idx * row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row = row - row_max
    row_exp = tl.exp(row)
    row_sum = tl.sum(row_exp, axis=0)
    row_softmax = row_exp / row_sum

    tl.store(out_start + col_offsets, row_softmax, mask=mask)


def _triton_softmax_last_dim(x: Tensor) -> Tensor:
    """Apply softmax over the last dimension using Triton."""
    orig_shape = x.shape
    # Reshape to 2D: (n_rows, n_cols)
    n_cols = orig_shape[-1]
    n_rows = x.numel() // n_cols

    x_2d = x.contiguous().view(n_rows, n_cols)
    output = torch.empty_like(x_2d)

    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)

    grid = (n_rows,)
    _softmax_kernel_2d[grid](
        x_2d,
        output,
        x_2d.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(orig_shape)


def softmax(input: Tensor, dim: int, dtype=None) -> Tensor:
    # Cast input if dtype is specified
    if dtype is not None:
        x = input.to(dtype)
    else:
        x = input

    # Only use Triton for CUDA floating-point tensors along the last dim
    # Normalize dim
    ndim = x.dim()
    normalized_dim = dim % ndim if ndim > 0 else dim

    if (
        x.is_cuda
        and x.is_floating_point()
        and not x.is_complex()
        and normalized_dim == ndim - 1
        and ndim >= 1
    ):
        try:
            result = _triton_softmax_last_dim(x)
            return result
        except Exception:
            pass

    # Fallback to PyTorch
    return F.softmax(input, dim=dim, dtype=dtype)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax(input: torch.Tensor, dim: int, dtype: torch.dtype=None) -> torch.Tensor:
#     return F.softmax(input, dim=dim, dtype=dtype)

def test_softmax():
    results = {}
    
    # Test case 1: Basic test with default dtype
    input1 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_1"] = softmax(input1, dim=1)
    
    # Test case 2: Test with different dimension
    input2 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_2"] = softmax(input2, dim=0)
    
    # Test case 3: Test with specified dtype
    input3 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_3"] = softmax(input3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with larger tensor
    input4 = torch.randn(100, 100, device='cuda')
    results["test_case_4"] = softmax(input4, dim=1)
    
    return results

test_results = test_softmax()
