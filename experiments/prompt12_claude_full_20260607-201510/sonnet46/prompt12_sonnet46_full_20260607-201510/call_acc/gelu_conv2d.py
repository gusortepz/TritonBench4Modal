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
    x_ptr,
    out_ptr,
    n_elements,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if approximate:
        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        coeff = 0.7978845608028654  # sqrt(2/pi)
        inner = coeff * (x + 0.044715 * x * x * x)
        tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        result = 0.5 * x * (1.0 + tanh_val)
    else:
        # exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))

    tl.store(out_ptr + offsets, result, mask=mask)


def _gelu_triton(x: Tensor, approximate: str = 'none') -> Tensor:
    if not x.is_cuda or not x.is_contiguous():
        return F.gelu(x, approximate=approximate)
    try:
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        use_tanh = approximate == 'tanh'
        _gelu_kernel[grid](
            x,
            out,
            n,
            use_tanh,
            BLOCK_SIZE,
        )
        return out
    except Exception:
        return F.gelu(x, approximate=approximate)


def _conv2d_gelu_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    approximate: str,
) -> Tensor:
    conv_out = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if conv_out.is_cuda and conv_out.dtype in (torch.float16, torch.float32, torch.bfloat16):
        conv_out_c = conv_out.contiguous()
        return _gelu_triton(conv_out_c, approximate=approximate)
    else:
        return F.gelu(conv_out, approximate=approximate)


try:
    _conv2d_gelu_fast = torch.compile(
        _conv2d_gelu_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _conv2d_gelu_fast = _conv2d_gelu_impl


def gelu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = 'none',
    out: Optional[Tensor] = None,
) -> Tensor:
    try:
        y = _conv2d_gelu_fast(input, weight, bias, stride, padding, dilation, groups, approximate)
    except Exception:
        y = _conv2d_gelu_impl(input, weight, bias, stride, padding, dilation, groups, approximate)

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
