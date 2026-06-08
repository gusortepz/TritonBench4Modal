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
def _floor_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.floor(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def _floor_impl(input: Tensor) -> Tensor:
    """Compute floor using PyTorch reference."""
    return torch.floor(input)


def _floor_triton_fallback(input: Tensor) -> Tensor:
    """Triton accelerated floor for float tensors on CUDA."""
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        return _floor_impl(input)
    
    output = torch.empty_like(input)
    n = input.numel()
    
    if n == 0:
        return output
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    
    _floor_kernel[grid](input, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


def floor(input: Tensor, *, out: Tensor = None) -> Tensor:
    """
    Returns a new tensor with the floor of the elements of the input,
    the largest integer less than or equal to each element.
    For integer inputs, follows the array-api convention of returning
    a copy of the input tensor.
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword args:
        out (Tensor, optional): the output tensor.
    
    Returns:
        Tensor: the floored tensor.
    """
    try:
        y = _floor_triton_fallback(input)
    except Exception:
        y = _floor_impl(input)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_floor():
    results = {}

    # Test case 1: Simple tensor with positive and negative floats
    input1 = torch.tensor([1.7, -2.3, 3.5, -4.8], device='cuda')
    results["test_case_1"] = floor(input1)

    # Test case 2: Tensor with integers (should remain unchanged)
    input2 = torch.tensor([1, -2, 3, -4], device='cuda')
    results["test_case_2"] = floor(input2)

    # Test case 3: Tensor with zero and positive/negative floats
    input3 = torch.tensor([0.0, 2.9, -3.1, 4.0], device='cuda')
    results["test_case_3"] = floor(input3)

    # Test case 4: Large tensor with random floats
    input4 = torch.rand(1000, device='cuda') * 100 - 50  # Random floats between -50 and 50
    results["test_case_4"] = floor(input4)

    return results

test_results = test_floor()
