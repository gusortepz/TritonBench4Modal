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
def _reciprocal_kernel(
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
    y = 1.0 / x
    tl.store(output_ptr + offsets, y, mask=mask)


def reciprocal(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Returns a new tensor with the reciprocal of the elements of the input.
    Integral inputs are promoted to the default scalar type.
    """
    
    # Promote integral inputs to default float type
    if input.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
        x = input.to(torch.float32)
    else:
        x = input
    
    # Use Triton fast path for CUDA float tensors
    if x.is_cuda and x.dtype in (torch.float32, torch.float64):
        y = torch.empty_like(x)
        n_elements = x.numel()
        
        if n_elements == 0:
            if out is not None:
                out.copy_(y)
                return out
            return y
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        
        try:
            _reciprocal_kernel[grid](
                x,
                y,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        except Exception:
            # Fallback to PyTorch if Triton fails
            y = torch.reciprocal(x)
    else:
        # PyTorch fallback for CPU or other dtypes
        y = torch.reciprocal(x)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_reciprocal():
    results = {}

    # Test case 1: Basic test with positive and negative numbers
    a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298], device='cuda')
    results["test_case_1"] = reciprocal(a)

    # Test case 2: Test with a tensor containing zero (expecting inf)
    b = torch.tensor([0.0, 1.0, -1.0, 2.0], device='cuda')
    results["test_case_2"] = reciprocal(b)

    # Test case 3: Test with a tensor containing large numbers
    c = torch.tensor([1e10, -1e10, 1e-10, -1e-10], device='cuda')
    results["test_case_3"] = reciprocal(c)

    # Test case 4: Test with a tensor of ones (expecting ones)
    d = torch.ones(4, device='cuda')
    results["test_case_4"] = reciprocal(d)

    return results

test_results = test_reciprocal()
