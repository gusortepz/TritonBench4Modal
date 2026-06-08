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
def _rsqrt_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.rsqrt(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def _triton_rsqrt(x: torch.Tensor) -> torch.Tensor:
    """Apply rsqrt elementwise using Triton kernel."""
    x_flat = x.contiguous().view(-1)
    out_flat = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    if n_elements == 0:
        return out_flat.view(x.shape)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _rsqrt_kernel[grid](
        x_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_flat.view(x.shape)


def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    """
    Returns the reciprocal of the square root of the tensordot product of a and b.
    
    Args:
        a: Left tensor to contract.
        b: Right tensor to contract.
        dims: Dimensions for contraction, as per torch.tensordot.
    
    Returns:
        Tensor: rsqrt(tensordot(a, b, dims))
    """
    # Compute tensordot using PyTorch (complex operation, not suitable for Triton)
    dot_result = torch.tensordot(a, b, dims=dims)

    # Apply rsqrt using Triton if on CUDA and floating point, else PyTorch fallback
    if (
        dot_result.is_cuda
        and dot_result.is_floating_point()
        and not dot_result.is_complex()
        and dot_result.numel() > 0
    ):
        try:
            # Ensure contiguous float tensor for Triton
            x = dot_result
            if x.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                return torch.rsqrt(x)
            # Triton works best with float32; cast if needed
            orig_dtype = x.dtype
            if orig_dtype == torch.float64:
                # Triton rsqrt may not support float64 on all hardware
                return torch.rsqrt(x)
            x_f32 = x.to(torch.float32) if orig_dtype == torch.bfloat16 else x
            result = _triton_rsqrt(x_f32.contiguous())
            if orig_dtype == torch.bfloat16:
                result = result.to(orig_dtype)
            return result
        except Exception:
            return torch.rsqrt(dot_result)
    else:
        return torch.rsqrt(dot_result)

##################################################################################################################################################



import torch

def test_tensordot_rsqrt():
    results = {}

    # Test case 1: Simple contraction with scalar result
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    dims = 1
    results["test_case_1"] = tensordot_rsqrt(a, b, dims)

    # Test case 2: Contraction with matrices
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_2"] = tensordot_rsqrt(a, b, dims)

    # Test case 3: Higher-dimensional tensors
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    b = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], device='cuda')
    dims = ([2], [1])
    results["test_case_3"] = tensordot_rsqrt(a, b, dims)

    # Test case 4: Different dimensions for contraction
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_4"] = tensordot_rsqrt(a, b, dims)

    return results

test_results = test_tensordot_rsqrt()
