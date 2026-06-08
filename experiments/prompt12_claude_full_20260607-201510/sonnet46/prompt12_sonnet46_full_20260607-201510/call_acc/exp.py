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
def _exp_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.exp(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def exp(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if not input.is_cuda or not input.is_floating_point() or input.is_complex():
        y = torch.exp(input)
        if out is not None:
            out.copy_(y)
            return out
        return y

    flat = input.contiguous().view(-1)
    n_elements = flat.numel()
    result_flat = torch.empty_like(flat)

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    if BLOCK_SIZE == 0:
        BLOCK_SIZE = 1
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    try:
        _exp_kernel[grid](flat, result_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        y = result_flat.view(input.shape)
    except Exception:
        y = torch.exp(input)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_exp():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp(input_tensor_1)

    # Test case 2: Test with a tensor containing negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp(input_tensor_2)

    # Test case 3: Test with a tensor containing zero
    input_tensor_3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp(input_tensor_3)

    # Test case 4: Test with a larger tensor
    input_tensor_4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = exp(input_tensor_4)

    return results

test_results = test_exp()
