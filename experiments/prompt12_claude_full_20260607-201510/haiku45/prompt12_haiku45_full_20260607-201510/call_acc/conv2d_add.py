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


def conv2d_add(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    other: Union[Tensor, float, int, None] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    alpha: Union[float, int] = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Applies 2D convolution followed by addition of a scaled tensor.
    
    Computes: conv2d(input, weight, bias, stride, padding, dilation, groups) + alpha * other
    """
    
    # Compute convolution result
    y = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    
    # Add other tensor or scalar if provided
    if other is not None:
        y = y + alpha * other
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



def test_conv2d_add():
    results = {}

    # Test case 1: Basic convolution with bias, no addition
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(2, device='cuda')
    results["test_case_1"] = conv2d_add(input_tensor, weight_tensor, bias=bias_tensor)

    # Test case 2: Convolution with addition of a scalar
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    scalar_addition = 2.0
    results["test_case_2"] = conv2d_add(input_tensor, weight_tensor, other=scalar_addition)

    # Test case 3: Convolution with addition of a tensor
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    results["test_case_3"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor)

    # Test case 4: Convolution with addition of a tensor and alpha scaling
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    alpha_value = 0.5
    results["test_case_4"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor, alpha=alpha_value)

    return results

test_results = test_conv2d_add()
