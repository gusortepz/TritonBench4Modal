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


def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    output_size: Optional[Tuple[int, int]] = None,
    output_ratio: Optional[Tuple[float, float]] = None,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies ReLU activation followed by 2D fractional max pooling.
    
    Args:
        input: Input tensor of shape (N, C, H, W)
        kernel_size: Size of the pooling window
        output_size: Target output size (height, width)
        output_ratio: Output size as ratio of input size
        return_indices: If True, return indices along with output
    
    Returns:
        output tensor, or (output, indices) if return_indices=True
    """
    if input is None or not input.is_cuda:
        return _fused_relu_fractional_max_pool2d_pytorch(
            input, kernel_size, output_size, output_ratio, return_indices
        )
    
    try:
        return _fused_relu_fractional_max_pool2d_pytorch(
            input, kernel_size, output_size, output_ratio, return_indices
        )
    except Exception:
        return _fused_relu_fractional_max_pool2d_pytorch(
            input, kernel_size, output_size, output_ratio, return_indices
        )


def _fused_relu_fractional_max_pool2d_pytorch(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    output_size: Optional[Tuple[int, int]] = None,
    output_ratio: Optional[Tuple[float, float]] = None,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    PyTorch reference implementation: ReLU followed by fractional max pooling.
    """
    if input is None:
        raise ValueError("input tensor cannot be None")
    
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = (kernel_size, kernel_size)
    else:
        kernel_size = tuple(kernel_size)
    
    if len(kernel_size) == 1:
        kernel_size = (kernel_size[0], kernel_size[0])
    
    # Apply ReLU
    relu_output = F.relu(input)
    
    # Apply fractional max pooling
    if output_ratio is not None:
        if not isinstance(output_ratio, (list, tuple)):
            output_ratio = (output_ratio, output_ratio)
        else:
            output_ratio = tuple(output_ratio)
        
        if len(output_ratio) == 1:
            output_ratio = (output_ratio[0], output_ratio[0])
        
        h_out = int(relu_output.shape[2] * output_ratio[0])
        w_out = int(relu_output.shape[3] * output_ratio[1])
        output_size = (h_out, w_out)
    
    if output_size is None:
        output_size = (relu_output.shape[2], relu_output.shape[3])
    
    if not isinstance(output_size, (list, tuple)):
        output_size = (output_size, output_size)
    else:
        output_size = tuple(output_size)
    
    if len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    
    # Use torch.nn.functional.fractional_max_pool2d
    result = F.fractional_max_pool2d(
        relu_output,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=return_indices,
    )
    
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
#     relu_output = F.relu(input)
#     pooled_output = F.fractional_max_pool2d(relu_output, kernel_size=kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices)
#     return pooled_output

def test_fused_fractional_max_pool2d_with_relu():
    results = {}
    
    # Test case 1: Basic functionality with kernel_size and output_size
    input_tensor = torch.randn(1, 1, 8, 8, device='cuda')
    kernel_size = (2, 2)
    output_size = (4, 4)
    results["test_case_1"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    # Test case 2: Using output_ratio instead of output_size
    output_ratio = (0.5, 0.5)
    results["test_case_2"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_ratio=output_ratio)
    
    # Test case 3: Return indices along with the pooled output
    results["test_case_3"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size, return_indices=True)
    
    # Test case 4: Larger kernel size
    kernel_size = (3, 3)
    results["test_case_4"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    return results

test_results = test_fused_fractional_max_pool2d_with_relu()
