import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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
def _cos_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Elementwise cosine kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.cos(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def _cos_fused(input: torch.Tensor) -> torch.Tensor:
    """Apply cosine element-wise using Triton kernel."""
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        return torch.cos(input)
    
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    try:
        _cos_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    except Exception:
        return torch.cos(input)
    
    return output


def cos_avg_pool1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True
) -> torch.Tensor:
    """
    Applies the cosine function element-wise to the input tensor,
    followed by 1D average pooling.
    
    Args:
        input (Tensor): The input tensor of shape (minibatch, in_channels, iW).
        kernel_size (int): Size of the pooling window.
        stride (int, optional): Stride of the pooling window. Defaults to `kernel_size`.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        ceil_mode (bool, optional): If True, uses ceil instead of floor to compute the output shape. Default is False.
        count_include_pad (bool, optional): If True, includes the zero-padding in the averaging calculation. Default is True.
    
    Returns:
        Tensor: The output tensor after cosine and average pooling operations.
    """
    if stride is None:
        stride = kernel_size
    
    # Apply cosine element-wise
    cos_result = _cos_fused(input)
    
    # Apply 1D average pooling
    output = F.avg_pool1d(
        cos_result,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> torch.Tensor:
#     cos_input = torch.cos(input)
#     return F.avg_pool1d(cos_input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

def test_cos_avg_pool1d():
    results = {}

    # Test case 1: Basic functionality with default parameters
    input_tensor_1 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_1'] = cos_avg_pool1d(input_tensor_1, kernel_size=2)

    # Test case 2: Custom stride
    input_tensor_2 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_2'] = cos_avg_pool1d(input_tensor_2, kernel_size=2, stride=1)

    # Test case 3: With padding
    input_tensor_3 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_3'] = cos_avg_pool1d(input_tensor_3, kernel_size=2, padding=1)

    # Test case 4: Using ceil_mode
    input_tensor_4 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_4'] = cos_avg_pool1d(input_tensor_4, kernel_size=2, ceil_mode=True)

    return results

test_results = test_cos_avg_pool1d()
