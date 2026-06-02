import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union


@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.sigmoid(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Perform the 2D convolution using PyTorch
    y = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Apply sigmoid activation element-wise
    if y.is_cuda and y.is_floating_point() and not y.is_complex():
        y_contig = y.contiguous()
        n = y_contig.numel()
        if n > 0:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            if out is not None:
                if out.is_cuda and out.is_floating_point() and not out.is_complex() and out.shape == y.shape:
                    out_contig = out.contiguous()
                    _sigmoid_kernel[grid](y_contig, out_contig, n, BLOCK_SIZE=1024)
                    out.copy_(out_contig)
                    return out
            y_sigmoid = torch.empty_like(y_contig)
            _sigmoid_kernel[grid](y_contig, y_sigmoid, n, BLOCK_SIZE=1024)
            return y_sigmoid

    # Fallback to PyTorch sigmoid for CPU, complex, integer, or unusual out cases
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
