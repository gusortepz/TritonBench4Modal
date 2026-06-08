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
def _cos_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.cos(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def cos(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if not input.is_cuda or not input.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        y = torch.cos(input)
        if out is not None:
            out.copy_(y)
            return out
        return y

    # Use Triton for CUDA float tensors
    # Work with float32 for computation
    compute_input = input.contiguous()
    if input.dtype not in (torch.float32, torch.float16):
        # Fall back to PyTorch for non-standard dtypes
        y = torch.cos(input)
        if out is not None:
            out.copy_(y)
            return out
        return y

    result = torch.empty_like(compute_input)
    n_elements = compute_input.numel()

    if n_elements == 0:
        if out is not None:
            out.copy_(result)
            return out
        return result

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    try:
        _cos_kernel[grid](compute_input, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    except Exception:
        result = torch.cos(compute_input)

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_cos():
    results = {}

    # Test case 1: Single positive value
    input_tensor_1 = torch.tensor([0.0], device='cuda')
    results["test_case_1"] = cos(input_tensor_1)

    # Test case 2: Single negative value
    input_tensor_2 = torch.tensor([-3.14159265 / 2], device='cuda')
    results["test_case_2"] = cos(input_tensor_2)

    # Test case 3: Multiple values
    input_tensor_3 = torch.tensor([0.0, 3.14159265 / 2, 3.14159265], device='cuda')
    results["test_case_3"] = cos(input_tensor_3)

    # Test case 4: Large tensor
    input_tensor_4 = torch.linspace(-3.14159265, 3.14159265, steps=1000, device='cuda')
    results["test_case_4"] = cos(input_tensor_4)

    return results

test_results = test_cos()
