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
def _gelu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    approximate: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if approximate == 0:
        cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
        y = x * cdf
    else:
        tanh_arg = 2.0 * (x + 0.044715 * x * x * x) * 0.7978845608
        tanh_result = 2.0 * tl.sigmoid(tanh_arg) - 1.0
        y = 0.5 * x * (1.0 + tanh_result)

    tl.store(output_ptr + offsets, y, mask=mask)


def gelu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = "none",
    out: Optional[Tensor] = None,
) -> Tensor:
    y = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    n_elements = y.numel()
    approximate_mode = 0 if approximate == "none" else 1

    if y.is_cuda and y.dtype in (torch.float32, torch.float64):
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        output = torch.empty_like(y)
        _gelu_kernel[grid](
            output,
            y,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            approximate=approximate_mode,
        )
        y = output
    else:
        y = F.gelu(y, approximate=approximate)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Tuple

# def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int], str]=0, dilation: Union[int, Tuple[int, int]]=1, groups: int=1, approximate: str='none', out: Optional[Tensor]=None) -> Tensor:
#     conv_result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.gelu(conv_result, approximate=approximate, out=out)

def test_gelu_conv2d():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = gelu_conv2d(input1, weight1)

    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = gelu_conv2d(input2, weight2, bias=bias2)

    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 8, 8, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = gelu_conv2d(input3, weight3, stride=2, padding=1)

    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 10, 10, device='cuda')
    weight4 = torch.randn(4, 1, 3, 3, device='cuda')
    results["test_case_4"] = gelu_conv2d(input4, weight4, dilation=2, groups=4)

    return results

test_results = test_gelu_conv2d()
