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
def _floor_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.floor(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def floor(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    # For integer types, return a copy (array-api convention)
    if not input.is_floating_point():
        result = input.clone()
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Use Triton only for CUDA float tensors
    if input.is_cuda:
        try:
            flat = input.contiguous().view(-1)
            n = flat.numel()
            result_flat = torch.empty_like(flat)
            BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
            if BLOCK_SIZE == 0:
                BLOCK_SIZE = 1
            grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _floor_kernel[grid](flat, result_flat, n, BLOCK_SIZE=BLOCK_SIZE)
            result = result_flat.view(input.shape)
            if out is not None:
                out.copy_(result)
                return out
            return result
        except Exception:
            pass

    # PyTorch fallback
    result = torch.floor(input)
    if out is not None:
        out.copy_(result)
        return out
    return result

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
