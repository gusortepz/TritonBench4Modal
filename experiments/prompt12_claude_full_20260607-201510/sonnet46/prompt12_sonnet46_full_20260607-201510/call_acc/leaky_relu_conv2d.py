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


@triton.jit
def _leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x >= 0.0, x, x * negative_slope)
    tl.store(out_ptr + offsets, result, mask=mask)


def _apply_leaky_relu_triton(x: Tensor, negative_slope: float, inplace: bool) -> Tensor:
    if not x.is_cuda or not x.is_contiguous():
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n, 1)))
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    try:
        _leaky_relu_kernel[grid](x, out, n, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
        return out
    except Exception:
        return F.leaky_relu(x, negative_slope=negative_slope, inplace=inplace)


def leaky_relu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> Tensor:
    # Apply 2D convolution
    conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding,
                        dilation=dilation, groups=groups)

    # Apply Leaky ReLU
    if conv_out.is_cuda and conv_out.dtype in (torch.float16, torch.float32, torch.bfloat16):
        conv_out = conv_out.contiguous()
        return _apply_leaky_relu_triton(conv_out, negative_slope, inplace)
    else:
        return F.leaky_relu(conv_out, negative_slope=negative_slope, inplace=inplace)

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def leaky_relu_conv2d(input: Tensor, weight: Tensor, bias: Tensor=None, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, negative_slope: float=0.01, inplace: bool=False) -> Tensor:
#     conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     output = F.leaky_relu(conv_output, negative_slope, inplace)
#     return output

def test_leaky_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_1"] = leaky_relu_conv2d(input, weight, bias)
    
    # Test case 2: Test with stride and padding
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_2"] = leaky_relu_conv2d(input, weight, bias, stride=2, padding=1)
    
    # Test case 3: Test with dilation and groups
    input = torch.randn(1, 6, 32, 32, device='cuda')
    weight = torch.randn(6, 1, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_3"] = leaky_relu_conv2d(input, weight, bias, dilation=2, groups=6)
    
    # Test case 4: Test with negative_slope and inplace
    input = torch.randn(1, 3, 32, 32, device='cuda')
    weight = torch.randn(6, 3, 3, 3, device='cuda')
    bias = torch.randn(6, device='cuda')
    results["test_case_4"] = leaky_relu_conv2d(input, weight, bias, negative_slope=0.1, inplace=True)
    
    return results

test_results = test_leaky_relu_conv2d()
