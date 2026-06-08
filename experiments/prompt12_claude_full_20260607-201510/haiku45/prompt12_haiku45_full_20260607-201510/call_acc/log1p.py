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
def _log1p_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(1.0 + x)
    
    tl.store(output_ptr + offsets, y, mask=mask)


def log1p(input: Tensor, *, out: Tensor = None) -> Tensor:
    """
    Returns a new tensor with the natural logarithm of (1 + input).
    This function is more accurate than torch.log for small values of input.
    """
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        y = torch.log1p(input)
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    y = torch.empty_like(input)
    n_elements = input.numel()
    
    if n_elements == 0:
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    
    _log1p_kernel[grid](input, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_log1p():
    results = {}

    # Test case 1: Basic test with a small positive tensor
    input1 = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = log1p(input1)

    # Test case 2: Test with a tensor containing zero
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    results["test_case_2"] = log1p(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-0.1, -0.2, -0.3], device='cuda')
    results["test_case_3"] = log1p(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = log1p(input4)

    return results

test_results = test_log1p()
