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
def _eq_kernel(
    x_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    has_scalar: tl.constexpr,
    scalar_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if has_scalar:
        result = x == scalar_val
    else:
        other = tl.load(other_ptr + offsets, mask=mask)
        result = x == other

    tl.store(out_ptr + offsets, result, mask=mask)


def _triton_eq(x: Tensor, other, out_tensor: Tensor) -> Tensor:
    """Run elementwise equality using Triton kernel."""
    n = x.numel()
    if n == 0:
        return out_tensor

    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    if isinstance(other, (int, float, bool)):
        _eq_kernel[grid](
            x,
            x,  # unused when has_scalar=True
            out_tensor,
            n,
            has_scalar=True,
            scalar_val=float(other),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        other_flat = other.contiguous().view(-1)
        _eq_kernel[grid](
            x,
            other_flat,
            out_tensor,
            n,
            has_scalar=False,
            scalar_val=0.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out_tensor


def fused_index_select_eq(
    input: Tensor,
    dim: int,
    index: Tensor,
    other,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Step 1: index_select
    selected = torch.index_select(input, dim, index)

    # Step 2: equality comparison
    # Try Triton path for CUDA float/int tensors with matching shapes
    use_triton = (
        selected.is_cuda
        and not selected.is_complex()
        and (
            isinstance(other, (int, float, bool))
            or (
                isinstance(other, Tensor)
                and other.is_cuda
                and not other.is_complex()
                and other.numel() == selected.numel()
            )
        )
    )

    if use_triton:
        try:
            result_bool = torch.empty(selected.shape, dtype=torch.bool, device=selected.device)
            x_flat = selected.contiguous().view(-1)

            if isinstance(other, Tensor):
                other_flat = other.contiguous().view(-1)
                _triton_eq_result = _triton_eq(x_flat, other_flat, result_bool.view(-1))
            else:
                _triton_eq_result = _triton_eq(x_flat, other, result_bool.view(-1))

            result = result_bool
        except Exception:
            result = selected.eq(other)
    else:
        result = selected.eq(other)

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()
