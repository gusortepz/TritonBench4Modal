import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tanh(x) = 2 * sigmoid(2x) - 1, computed in a stable way via exp
    # Use 1 - 2 / (exp(2x) + 1) for numerical stability via sign trick
    # Simpler: use the direct formula via tl.exp
    # Stable tanh: clamp 2x to avoid overflow
    two_x = 2.0 * x
    # exp(2x) can overflow; use sign trick
    # tanh(x) = sign(x) * (1 - 2 / (exp(2|x|) + 1))
    abs_two_x = tl.abs(two_x)
    e = tl.exp(-abs_two_x)
    t = (1.0 - e) / (1.0 + e)
    # Apply sign of x
    sign = tl.where(x >= 0, 1.0, -1.0)
    y = sign * t
    tl.store(y_ptr + offsets, y, mask=mask)


def tanh(input, *, out=None):
    # Use PyTorch fallback for non-CUDA, complex, integer, or non-floating cases
    if (not isinstance(input, torch.Tensor)
            or not input.is_cuda
            or input.dtype not in (torch.float16, torch.float32, torch.bfloat16)
            or input.numel() == 0):
        return torch.tanh(input, out=out)

    x = input.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _tanh_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_tanh():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5], device='cuda')
    results["test_case_1"] = tanh(input_tensor_1)

    # Test case 2: Test with a 2D tensor
    input_tensor_2 = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], device='cuda')
    results["test_case_2"] = tanh(input_tensor_2)

    # Test case 3: Test with a larger tensor
    input_tensor_3 = torch.randn(100, 100, device='cuda')
    results["test_case_3"] = tanh(input_tensor_3)

    # Test case 4: Test with an empty tensor
    input_tensor_4 = torch.tensor([], device='cuda')
    results["test_case_4"] = tanh(input_tensor_4)

    return results

test_results = test_tanh()
