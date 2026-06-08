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
def _tanh_kernel(
    y_ptr,
    y_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    In-place tanh activation kernel.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets * y_stride, mask=mask, other=0.0)
    y_tanh = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    tl.store(y_ptr + offsets * y_stride, y_tanh, mask=mask)


def tanh_linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Applies a linear transformation followed by Tanh activation.
    
    Args:
        input (Tensor): Input tensor of shape (*, in_features).
        weight (Tensor): Weight matrix of shape (out_features, in_features).
        bias (Tensor, optional): Bias tensor of shape (out_features). Default: None.
    
    Returns:
        Tensor: Output tensor of shape (*, out_features) with Tanh applied.
    """
    # PyTorch reference path: linear transformation + tanh
    y = F.linear(input, weight, bias)
    
    # Check if input is CUDA and floating-point for Triton path
    if y.is_cuda and y.dtype in [torch.float32, torch.float64, torch.float16]:
        # Flatten to 1D for Triton kernel, then reshape back
        original_shape = y.shape
        y_flat = y.reshape(-1)
        n_elements = y_flat.numel()
        
        # Determine block size: cap at 1024 to avoid OOM
        block_size = min(triton.next_power_of_2(n_elements), 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        
        # Create output tensor (in-place modification)
        y_out = y_flat.contiguous()
        
        try:
            _tanh_kernel[grid](
                y_out,
                y_out.stride(0),
                n_elements,
                BLOCK_SIZE=block_size,
            )
            return y_out.reshape(original_shape)
        except Exception:
            # Fallback to PyTorch tanh if Triton fails
            return torch.tanh(y)
    else:
        # Fallback for non-CUDA or unsupported dtypes
        return torch.tanh(y)

##################################################################################################################################################



import torch
from tanh_linear import tanh_linear

def test_tanh_linear():
    results = {}

    # Test case 1: input, weight, and bias on GPU
    input1 = torch.randn(5, 3, device='cuda')
    weight1 = torch.randn(4, 3, device='cuda')
    bias1 = torch.randn(4, device='cuda')
    result1 = tanh_linear(input1, weight1, bias1)
    results["test_case_1"] = result1

    # Test case 2: input and weight on GPU, bias is None
    input2 = torch.randn(5, 3, device='cuda')
    weight2 = torch.randn(4, 3, device='cuda')
    result2 = tanh_linear(input2, weight2)
    results["test_case_2"] = result2

    # Test case 3: input and weight on GPU, bias on GPU
    input3 = torch.randn(2, 3, device='cuda')
    weight3 = torch.randn(2, 3, device='cuda')
    bias3 = torch.randn(2, device='cuda')
    result3 = tanh_linear(input3, weight3, bias3)
    results["test_case_3"] = result3

    # Test case 4: input, weight, and bias on GPU with different dimensions
    input4 = torch.randn(3, 2, device='cuda')
    weight4 = torch.randn(2, 2, device='cuda')
    bias4 = torch.randn(2, device='cuda')
    result4 = tanh_linear(input4, weight4, bias4)
    results["test_case_4"] = result4

    return results

test_results = test_tanh_linear()
