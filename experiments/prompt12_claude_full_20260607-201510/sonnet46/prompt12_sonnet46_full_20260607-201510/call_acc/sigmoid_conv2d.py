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
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.sigmoid(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def _apply_sigmoid_triton(x: Tensor) -> Tensor:
    if not x.is_cuda or not x.is_contiguous():
        return torch.sigmoid(x)
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _sigmoid_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def sigmoid_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Step 1: apply conv2d
    conv_out = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Step 2: apply sigmoid
    if conv_out.is_cuda:
        # Make contiguous for Triton kernel
        conv_out_c = conv_out.contiguous()
        y = _apply_sigmoid_triton(conv_out_c)
    else:
        y = torch.sigmoid(conv_out)

    # Handle out parameter
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
