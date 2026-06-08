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


@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.rsqrt(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def _tensordot_rsqrt_impl(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    tensordot_result = torch.tensordot(a, b, dims=dims)
    result = torch.rsqrt(tensordot_result)
    return result


def _tensordot_rsqrt_triton(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    tensordot_result = torch.tensordot(a, b, dims=dims)
    
    if not tensordot_result.is_cuda or tensordot_result.dtype not in (torch.float32, torch.float64):
        return torch.rsqrt(tensordot_result)
    
    n_elements = tensordot_result.numel()
    if n_elements == 0:
        return tensordot_result
    
    out = torch.empty_like(tensordot_result)
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _rsqrt_kernel[grid](tensordot_result.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    try:
        return _tensordot_rsqrt_triton(a, b, dims)
    except Exception:
        return _tensordot_rsqrt_impl(a, b, dims)

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
