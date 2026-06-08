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

ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805

@triton.jit
def _selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(0.0, ALPHA * (tl.exp(x) - 1.0))
    result = SCALE * (pos + neg)
    tl.store(out_ptr + offsets, result, mask=mask)


def _selu_triton(input: Tensor) -> Tensor:
    out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _selu_kernel[grid](
        input.contiguous(),
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        try:
            if inplace:
                result = _selu_triton(input)
                input.copy_(result)
                return input
            else:
                return _selu_triton(input)
        except Exception:
            pass
    return F.selu(input, inplace=inplace)

##################################################################################################################################################



def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    return results

test_results = test_selu()
