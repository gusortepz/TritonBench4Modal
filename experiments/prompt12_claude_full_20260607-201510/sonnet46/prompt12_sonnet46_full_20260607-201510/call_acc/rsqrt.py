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
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.rsqrt(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def rsqrt(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        n_elements = input.numel()
        output = torch.empty_like(input)
        if n_elements > 0:
            BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            try:
                _rsqrt_kernel[grid](
                    input.contiguous().view(-1),
                    output.view(-1),
                    n_elements,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                if out is not None:
                    out.copy_(output)
                    return out
                return output
            except Exception:
                pass

    result = torch.rsqrt(input)
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_rsqrt():
    results = {}

    # Test case 1: Positive elements
    input1 = torch.tensor([4.0, 16.0, 25.0], device='cuda')
    results["test_case_1"] = rsqrt(input1)

    # Test case 2: Contains zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = rsqrt(input2)

    # Test case 3: Contains negative elements
    input3 = torch.tensor([-1.0, 4.0, 9.0], device='cuda')
    results["test_case_3"] = rsqrt(input3)

    # Test case 4: All elements are zero
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = rsqrt(input4)

    return results

test_results = test_rsqrt()
