import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union


@triton.jit
def _sigmoid_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sigmoid(x)
    tl.store(x_ptr + offsets, y, mask=mask)


def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1, out=None):
    """
    Applies a 2D convolution over an input tensor followed by element-wise
    sigmoid activation.
    """
    # Perform the convolution using PyTorch's functional API (correctness first).
    y = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Apply sigmoid: use Triton when safe (CUDA, floating-point, contiguous),
    # otherwise fall back to torch.sigmoid.
    use_triton = (
        y.is_cuda
        and y.is_floating_point()
        and y.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )

    if use_triton:
        y_contig = y.contiguous()
        n_elements = y_contig.numel()
        if n_elements > 0:
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _sigmoid_inplace_kernel[grid](y_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        y = y_contig
    else:
        y = torch.sigmoid(y)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
#     conv_result = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     result = torch.sigmoid(conv_result)
#     return result

def test_sigmoid_conv2d():
    results = {}

    # Test case 1: Basic test with no bias, stride, padding, dilation, or groups
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = sigmoid_conv2d(input1, weight1)

    # Test case 2: Test with bias
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = sigmoid_conv2d(input1, weight1, bias=bias2)

    # Test case 3: Test with stride
    results["test_case_3"] = sigmoid_conv2d(input1, weight1, stride=2)

    # Test case 4: Test with padding
    results["test_case_4"] = sigmoid_conv2d(input1, weight1, padding=1)

    return results

test_results = test_sigmoid_conv2d()
