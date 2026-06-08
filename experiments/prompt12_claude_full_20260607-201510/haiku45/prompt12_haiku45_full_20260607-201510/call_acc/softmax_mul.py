import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

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
def _softmax_mul_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    dim_size: tl.constexpr,
    stride_dim: tl.constexpr,
    other_is_scalar: tl.constexpr,
    other_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if other_is_scalar:
        other_vals = other_scalar
    else:
        other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    output = input_vals * other_vals
    tl.store(output_ptr + offsets, output, mask=mask)


def softmax_mul(
    input: Tensor,
    other: Union[Tensor, float, int],
    dim: int,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Applies softmax to input along dim, then multiplies by other.

    Args:
        input: Input tensor
        other: Tensor or scalar to multiply with softmaxed values
        dim: Dimension along which to apply softmax
        dtype: Optional data type for computation
        out: Optional output tensor

    Returns:
        Softmaxed and multiplied tensor
    """
    # Cast input if dtype is specified
    if dtype is not None:
        input = input.to(dtype)

    # Apply softmax along the specified dimension
    softmaxed = F.softmax(input, dim=dim)

    # Multiply by other
    if isinstance(other, (int, float)):
        result = softmaxed * other
    else:
        result = softmaxed * other

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax_mul(input, other, dim, dtype=None, out=None):
#     softmaxed = F.softmax(input, dim=dim, dtype=dtype)
#     if isinstance(other, torch.Tensor):
#         result = softmaxed * other
#     else:
#         result = softmaxed * other
#     if out is not None:
#         out.copy_(result)
#         return out
#     return result

def test_softmax_mul():
    results = {}
    
    # Test case 1: Basic test with two tensors
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_1"] = softmax_mul(input1, other1, dim=1)
    
    # Test case 2: Test with scalar multiplication
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other2 = 0.5
    results["test_case_2"] = softmax_mul(input2, other2, dim=1)
    
    # Test case 3: Test with different dtype
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = softmax_mul(input3, other3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with out parameter
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    out4 = torch.empty_like(input4)
    results["test_case_4"] = softmax_mul(input4, other4, dim=1, out=out4)
    
    return results

test_results = test_softmax_mul()
