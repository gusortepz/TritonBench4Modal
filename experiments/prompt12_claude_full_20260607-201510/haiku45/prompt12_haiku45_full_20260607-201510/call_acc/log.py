import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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
def _log_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.log(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def log(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if not input.is_cuda or input.dtype in (torch.complex64, torch.complex128):
        y = torch.log(input)
    else:
        n = input.numel()
        if n == 0:
            y = torch.log(input)
        else:
            output = torch.empty_like(input)
            BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
            _log_kernel[grid](input, output, n, BLOCK_SIZE=BLOCK_SIZE)
            y = output
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_log():
    results = {}

    # Test case 1: Basic test with positive values
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = log(input1)

    # Test case 2: Test with a tensor containing a zero
    input2 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_2"] = log(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = log(input3)

    # Test case 4: Test with a tensor containing a mix of positive, negative, and zero
    input4 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_4"] = log(input4)

    return results

test_results = test_log()
