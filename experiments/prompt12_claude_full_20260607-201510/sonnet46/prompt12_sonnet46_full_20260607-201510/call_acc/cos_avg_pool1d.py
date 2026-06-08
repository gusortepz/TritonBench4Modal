import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
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


# Triton kernel for elementwise cosine
@triton.jit
def _cos_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # cos(x) = sin(x + pi/2)
    # Use tl.cos if available, otherwise use identity: cos(x) = 1 - 2*sin^2(x/2)
    # tl supports tl.cos via libdevice equivalent
    out = tl.cos(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def _cos_triton(input: Tensor) -> Tensor:
    """Apply cosine elementwise using Triton kernel."""
    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _cos_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _cos_avg_pool1d_impl(
    input: Tensor,
    kernel_size: int,
    stride: Optional[int],
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
) -> Tensor:
    # Elementwise cos
    cosined = torch.cos(input)
    # 1D average pooling
    effective_stride = stride if stride is not None else kernel_size
    return F.avg_pool1d(
        cosined,
        kernel_size=kernel_size,
        stride=effective_stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )


try:
    _cos_avg_pool1d_fast = torch.compile(
        _cos_avg_pool1d_impl, mode="max-autotune", fullgraph=False
    )
except Exception:
    _cos_avg_pool1d_fast = _cos_avg_pool1d_impl


def cos_avg_pool1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> torch.Tensor:
    """
    Applies cosine element-wise to the input tensor, then applies 1D average pooling.

    Args:
        input (Tensor): Input tensor of shape (minibatch, in_channels, iW).
        kernel_size (int): Size of the pooling window.
        stride (int, optional): Stride of the pooling window. Defaults to kernel_size.
        padding (int, optional): Zero-padding added to both sides. Default is 0.
        ceil_mode (bool, optional): If True, uses ceil for output shape. Default is False.
        count_include_pad (bool, optional): If True, includes zero-padding in averaging. Default is True.

    Returns:
        Tensor: Output after applying cos and avg_pool1d.
    """
    # Use Triton for cos when on CUDA with float types, then pooling in PyTorch
    if (
        input.is_cuda
        and input.is_contiguous()
        and input.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and input.numel() > 0
    ):
        try:
            # Apply Triton cos kernel
            cosined = _cos_triton(input)
            effective_stride = stride if stride is not None else kernel_size
            return F.avg_pool1d(
                cosined,
                kernel_size=kernel_size,
                stride=effective_stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        except Exception:
            pass

    # Fallback: compiled impl or reference
    try:
        return _cos_avg_pool1d_fast(
            input, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
    except Exception:
        return _cos_avg_pool1d_impl(
            input, kernel_size, stride, padding, ceil_mode, count_include_pad
        )

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
