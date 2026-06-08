import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

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
def _exp_mean_kernel(
    input_ptr,
    output_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x)
    
    tl.store(output_ptr + offsets, y, mask=mask)


def _exp_mean_flat(input: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Compute exp(input).mean() for flat tensor."""
    numel = input.numel()
    
    if numel == 0:
        result = torch.tensor(0.0, dtype=dtype or input.dtype, device=input.device)
        return result
    
    BLOCK_SIZE = min(triton.next_power_of_2(numel), 1024)
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    exp_result = torch.empty(numel, dtype=input.dtype, device=input.device)
    
    _exp_mean_kernel[grid](input.view(-1), exp_result, numel, BLOCK_SIZE=BLOCK_SIZE)
    
    mean_val = exp_result.mean(dtype=dtype)
    return mean_val


def exp_mean(
    input: Tensor,
    dim: Optional[Union[int, tuple]] = None,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Applies the exponential function to each element in the input tensor
    and then computes the mean value along the specified dimension or over all elements.
    
    Args:
        input: Input tensor.
        dim: Dimension along which to compute the mean. If None, mean over all elements.
        keepdim: Whether to keep the reduced dimension.
        dtype: Output dtype. If None, uses input dtype.
        out: Optional output tensor.
    
    Returns:
        Tensor with exponential values averaged along the specified dimension.
    """
    if input.is_cuda and input.dtype in (torch.float32, torch.float64):
        if dim is None:
            y = _exp_mean_flat(input, dtype=dtype)
        else:
            exp_input = torch.exp(input)
            y = torch.mean(exp_input, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        exp_input = torch.exp(input)
        y = torch.mean(exp_input, dim=dim, keepdim=keepdim, dtype=dtype)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()
