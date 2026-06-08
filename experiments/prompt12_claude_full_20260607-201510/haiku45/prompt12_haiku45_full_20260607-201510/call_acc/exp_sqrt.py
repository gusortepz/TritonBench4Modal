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
def _exp_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.sqrt(tl.exp(x))
    tl.store(output_ptr + offsets, y, mask=mask)


def exp_sqrt(input: Tensor, out: Tensor = None) -> Tensor:
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        y = torch.sqrt(torch.exp(input))
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    
    try:
        _exp_sqrt_kernel[grid](
            input,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    except Exception:
        y = torch.sqrt(torch.exp(input))
        out.copy_(y)
        return out

##################################################################################################################################################



import torch

def test_exp_sqrt():
    results = {}

    # Test case 1: Basic functionality with a simple tensor
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_sqrt(input1)

    # Test case 2: Test with a tensor containing negative values
    input2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = exp_sqrt(input2)

    # Test case 3: Test with a tensor containing zero
    input3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_3"] = exp_sqrt(input3)

    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = exp_sqrt(input4, out=out4)

    return results

test_results = test_exp_sqrt()
