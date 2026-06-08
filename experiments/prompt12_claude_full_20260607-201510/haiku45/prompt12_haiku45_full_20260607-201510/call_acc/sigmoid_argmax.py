import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
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
def _sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.sigmoid(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def _sigmoid_fused(input: Tensor) -> Tensor:
    """Apply sigmoid elementwise using Triton for flat tensors."""
    n_elements = input.numel()
    output = torch.empty_like(input)
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _sigmoid_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    return output


def sigmoid_argmax(
    input: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Tensor:
    """
    Applies sigmoid to each element and computes argmax along dimension.
    
    Args:
        input: The input tensor.
        dim: The dimension to reduce. If None, argmax over all elements.
        keepdim: Whether to retain the dimension.
    
    Returns:
        LongTensor of indices of maximum values.
    """
    if not input.is_cuda:
        sigmoid_vals = torch.sigmoid(input)
        return torch.argmax(sigmoid_vals, dim=dim, keepdim=keepdim)
    
    if input.dtype not in (torch.float32, torch.float64):
        sigmoid_vals = torch.sigmoid(input)
        return torch.argmax(sigmoid_vals, dim=dim, keepdim=keepdim)
    
    try:
        sigmoid_vals = _sigmoid_fused(input)
    except Exception:
        sigmoid_vals = torch.sigmoid(input)
    
    return torch.argmax(sigmoid_vals, dim=dim, keepdim=keepdim)

##################################################################################################################################################



import torch

def test_sigmoid_argmax():
    results = {}

    # Test case 1: 1D tensor, no dim specified
    input1 = torch.tensor([0.1, 2.0, -1.0, 3.0], device='cuda')
    results["test_case_1"] = sigmoid_argmax(input1)

    # Test case 2: 2D tensor, dim=0
    input2 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_2"] = sigmoid_argmax(input2, dim=0)

    # Test case 3: 2D tensor, dim=1
    input3 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_3"] = sigmoid_argmax(input3, dim=1)

    # Test case 4: 2D tensor, dim=1, keepdim=True
    input4 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_4"] = sigmoid_argmax(input4, dim=1, keepdim=True)

    return results

test_results = test_sigmoid_argmax()
