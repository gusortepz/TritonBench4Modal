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
def _tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tanh via sigmoid: tanh(x) = 2*sigmoid(2*x) - 1
    result = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    tl.store(out_ptr + offsets, result, mask=mask)


def _tanh_triton(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def tanh_linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    # Apply linear transformation
    linear_out = F.linear(input, weight, bias)

    # Apply Tanh activation
    if linear_out.is_cuda and linear_out.is_contiguous() and linear_out.dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            flat = linear_out.contiguous().view(-1)
            result = _tanh_triton(flat)
            return result.view(linear_out.shape)
        except Exception:
            return torch.tanh(linear_out)
    else:
        return torch.tanh(linear_out)

##################################################################################################################################################



import torch
from tanh_linear import tanh_linear

def test_tanh_linear():
    results = {}

    # Test case 1: input, weight, and bias on GPU
    input1 = torch.randn(5, 3, device='cuda')
    weight1 = torch.randn(4, 3, device='cuda')
    bias1 = torch.randn(4, device='cuda')
    result1 = tanh_linear(input1, weight1, bias1)
    results["test_case_1"] = result1

    # Test case 2: input and weight on GPU, bias is None
    input2 = torch.randn(5, 3, device='cuda')
    weight2 = torch.randn(4, 3, device='cuda')
    result2 = tanh_linear(input2, weight2)
    results["test_case_2"] = result2

    # Test case 3: input and weight on GPU, bias on GPU
    input3 = torch.randn(2, 3, device='cuda')
    weight3 = torch.randn(2, 3, device='cuda')
    bias3 = torch.randn(2, device='cuda')
    result3 = tanh_linear(input3, weight3, bias3)
    results["test_case_3"] = result3

    # Test case 4: input, weight, and bias on GPU with different dimensions
    input4 = torch.randn(3, 2, device='cuda')
    weight4 = torch.randn(2, 2, device='cuda')
    bias4 = torch.randn(2, device='cuda')
    result4 = tanh_linear(input4, weight4, bias4)
    results["test_case_4"] = result4

    return results

test_results = test_tanh_linear()
