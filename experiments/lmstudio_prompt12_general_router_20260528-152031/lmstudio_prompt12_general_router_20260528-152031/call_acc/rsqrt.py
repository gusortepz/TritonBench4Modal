import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

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
def _rsqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output = tl.rsqrt(x)
    tl.store(output_ptr + offsets, output, mask=mask)

def rsqrt(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is not None:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.dtype == input.dtype, "Output dtype must match input dtype"
    
    if input.numel() == 0:
        if out is not None:
            return out
        return torch.empty_like(input)
    
    if out is not None:
        output = out
    else:
        output = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    while BLOCK_SIZE < n_elements:
        BLOCK_SIZE *= 2
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _rsqrt_kernel[grid](input, output, n_elements, BLOCK_SIZE)
    
    return output

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
