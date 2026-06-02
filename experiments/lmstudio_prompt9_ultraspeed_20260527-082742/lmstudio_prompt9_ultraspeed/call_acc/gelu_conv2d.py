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

def _gelu_conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, approximate='none'):
    conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return F.gelu(conv_out, approximate=approximate)

try:
    _gelu_conv2d_fast = torch.compile(_gelu_conv2d_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _gelu_conv2d_fast = _gelu_conv2d_impl

def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    y = _gelu_conv2d_fast(input, weight, bias, stride, padding, dilation, groups, approximate)
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
