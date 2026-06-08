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


def _log_softmax_linear_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    dim: int,
    dtype: Optional[torch.dtype],
) -> Tensor:
    # Cast input if dtype is specified
    x = input
    if dtype is not None:
        x = x.to(dtype)
        weight = weight.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)

    # Linear transformation: (*, in_features) @ (in_features, out_features) -> (*, out_features)
    out = F.linear(x, weight, bias)

    # Apply log_softmax along the specified dimension
    out = F.log_softmax(out, dim=dim)

    return out


try:
    _log_softmax_linear_fast = torch.compile(
        _log_softmax_linear_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _log_softmax_linear_fast = _log_softmax_linear_impl


def log_softmax_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    dim: int = -1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Applies a linear transformation to the input tensor followed by the
    log_softmax activation function.

    Args:
        input: The input tensor of shape (*, in_features).
        weight: The weight matrix of shape (out_features, in_features).
        bias: The optional bias tensor of shape (out_features). Default: None.
        dim: The dimension along which log_softmax will be computed. Default: -1.
        dtype: The desired data type of the returned tensor. Default: None.

    Returns:
        Tensor after linear transformation and log_softmax.
    """
    try:
        return _log_softmax_linear_fast(input, weight, bias, dim, dtype)
    except Exception:
        return _log_softmax_linear_impl(input, weight, bias, dim, dtype)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
#     output = torch.matmul(input, weight.T)
#     if bias is not None:
#         output += bias
#     return F.log_softmax(output, dim=dim, dtype=dtype)

def test_log_softmax_linear():
    results = {}

    # Test case 1: Basic test with bias
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    bias1 = torch.tensor([0.1, -0.1], device='cuda')
    results["test_case_1"] = log_softmax_linear(input1, weight1, bias1)

    # Test case 2: Test without bias
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    results["test_case_2"] = log_softmax_linear(input2, weight2)

    # Test case 3: Test with different dim
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight3 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    bias3 = torch.tensor([0.1, -0.1], device='cuda')
    results["test_case_3"] = log_softmax_linear(input3, weight3, bias3, dim=0)

    # Test case 4: Test with dtype
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, 0.5], [0.5, -0.5]], device='cuda')
    bias4 = torch.tensor([0.1, -0.1], device='cuda')
    results["test_case_4"] = log_softmax_linear(input4, weight4, bias4, dtype=torch.float64)

    return results

test_results = test_log_softmax_linear()
