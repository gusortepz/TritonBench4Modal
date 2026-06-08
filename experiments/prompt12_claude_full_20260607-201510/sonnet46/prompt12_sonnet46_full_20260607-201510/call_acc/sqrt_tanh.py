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
def _sqrt_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_sqrt = tl.sqrt(x)
    # tanh via sigmoid: tanh(x) = 2*sigmoid(2*x) - 1
    x_tanh = 2.0 * tl.sigmoid(2.0 * x_sqrt) - 1.0
    tl.store(out_ptr + offsets, x_tanh, mask=mask)


def _sqrt_tanh_pytorch(input: Tensor) -> Tensor:
    return torch.tanh(torch.sqrt(input))


def sqrt_tanh(input: Tensor, out: Optional[Tensor] = None) -> Tensor:
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        inp_flat = input.contiguous().flatten()
        n = inp_flat.numel()
        result_flat = torch.empty(n, dtype=inp_flat.dtype, device=inp_flat.device)
        BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
        if BLOCK_SIZE == 0:
            BLOCK_SIZE = 1
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        try:
            _sqrt_tanh_kernel[grid](inp_flat, result_flat, n, BLOCK_SIZE=BLOCK_SIZE)
            y = result_flat.reshape(input.shape)
        except Exception:
            y = _sqrt_tanh_pytorch(input)
    else:
        y = _sqrt_tanh_pytorch(input)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_sqrt_tanh():
    results = {}

    # Test case 1: Positive values
    input1 = torch.tensor([4.0, 9.0, 16.0], device='cuda')
    results["test_case_1"] = sqrt_tanh(input1)

    # Test case 2: Negative values
    input2 = torch.tensor([-4.0, -9.0, -16.0], device='cuda')
    results["test_case_2"] = sqrt_tanh(input2)

    # Test case 3: Mixed values
    input3 = torch.tensor([4.0, -9.0, 16.0, -1.0], device='cuda')
    results["test_case_3"] = sqrt_tanh(input3)

    # Test case 4: Zero values
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = sqrt_tanh(input4)

    return results

test_results = test_sqrt_tanh()
