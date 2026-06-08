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
def _exp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def _exp_triton(input: Tensor) -> Tensor:
    """Apply exp elementwise using Triton kernel."""
    output = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return output
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _exp_kernel[grid](
        input.contiguous().view(-1),
        output.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def _exp_mean_impl(input: Tensor, dim=None, keepdim: bool = False, dtype=None) -> Tensor:
    # Apply exp then mean
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        exp_result = _exp_triton(input.contiguous())
    else:
        exp_result = torch.exp(input)

    if dtype is not None:
        exp_result = exp_result.to(dtype)

    if dim is None:
        result = exp_result.mean()
        if keepdim:
            result = result.reshape([1] * input.dim())
    else:
        result = exp_result.mean(dim=dim, keepdim=keepdim)

    return result


def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> Tensor:
    y = _exp_mean_impl(input, dim=dim, keepdim=keepdim, dtype=dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()
