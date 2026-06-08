import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Union, Tuple

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
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.sigmoid(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def _sigmoid_elementwise(x: Tensor) -> Tensor:
    """Apply sigmoid element-wise using Triton."""
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return torch.sigmoid(x)
    
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _sigmoid_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


def sigmoid_adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """
    Applies 2D adaptive average pooling followed by sigmoid activation.
    
    Args:
        input: Input tensor of shape (N, C, H, W)
        output_size: Target output spatial dimensions (single int or tuple of ints)
    
    Returns:
        Tensor of shape (N, C, OH, OW) where OH, OW are determined by output_size
    """
    # Validate input
    if input.dim() != 4:
        raise ValueError(f"Expected 4D input, got {input.dim()}D")
    
    # Normalize output_size
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (list, tuple)):
        output_size = tuple(output_size)
        if len(output_size) != 2:
            raise ValueError(f"output_size must have 2 elements, got {len(output_size)}")
    else:
        raise TypeError(f"output_size must be int or tuple, got {type(output_size)}")
    
    # Apply adaptive average pooling
    pooled = F.adaptive_avg_pool2d(input, output_size)
    
    # Apply sigmoid activation
    result = _sigmoid_elementwise(pooled)
    
    return result

##################################################################################################################################################



def test_sigmoid_adaptive_avg_pool2d():
    # Initialize a dictionary to store the results of each test case
    results = {}

    # Test case 1: Basic test with a 4D tensor and output size as an integer
    input_tensor1 = torch.randn(1, 3, 8, 8, device='cuda')  # Batch size 1, 3 channels, 8x8 size
    output_size1 = 4
    result1 = sigmoid_adaptive_avg_pool2d(input_tensor1, output_size1)
    results["test_case_1"] = result1

    # Test case 2: Test with a 4D tensor and output size as a tuple
    input_tensor2 = torch.randn(2, 3, 10, 10, device='cuda')  # Batch size 2, 3 channels, 10x10 size
    output_size2 = (5, 5)
    result2 = sigmoid_adaptive_avg_pool2d(input_tensor2, output_size2)
    results["test_case_2"] = result2

    # Test case 3: Test with a larger batch size
    input_tensor3 = torch.randn(4, 3, 16, 16, device='cuda')  # Batch size 4, 3 channels, 16x16 size
    output_size3 = (8, 8)
    result3 = sigmoid_adaptive_avg_pool2d(input_tensor3, output_size3)
    results["test_case_3"] = result3

    # Test case 4: Test with a single channel
    input_tensor4 = torch.randn(1, 1, 12, 12, device='cuda')  # Batch size 1, 1 channel, 12x12 size
    output_size4 = (6, 6)
    result4 = sigmoid_adaptive_avg_pool2d(input_tensor4, output_size4)
    results["test_case_4"] = result4

    return results

test_results = test_sigmoid_adaptive_avg_pool2d()
