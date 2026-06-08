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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # sigmoid(x) = 1 / (1 + exp(-x))
    result = tl.sigmoid(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def _sigmoid_triton(x: Tensor) -> Tensor:
    if not x.is_cuda or not x.is_contiguous():
        return torch.sigmoid(x)
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    try:
        _sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
        return out
    except Exception:
        return torch.sigmoid(x)


def dropout_sigmoid_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    p=0.5,
    training=True,
    inplace=False,
) -> torch.Tensor:
    # Step 1: Linear transformation
    x = F.linear(input, weight, bias)

    # Step 2: Sigmoid activation
    if x.is_cuda and x.is_contiguous():
        x = _sigmoid_triton(x)
    else:
        x = torch.sigmoid(x)

    # Step 3: Dropout
    x = F.dropout(x, p=p, training=training, inplace=inplace)

    return x

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
#     """
#     Applies a linear transformation followed by a sigmoid activation and dropout.

#     Args:
#         input (torch.Tensor): Input tensor of shape (*, in_features).
#         weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
#         bias (torch.Tensor, optional): Bias tensor of shape (out_features). Default: None.
#         p (float, optional): Probability of an element to be zeroed in dropout. Default: 0.5.
#         training (bool, optional): If True, applies dropout during training. Default: True.
#         inplace (bool, optional): If True, performs the operation in-place. Default: False.

#     Returns:
#         torch.Tensor: The resulting tensor after applying the linear transformation, sigmoid activation, and dropout.
#     """
#     output = F.linear(input, weight, bias)
#     output = torch.sigmoid(output)
#     if training:
#         output = F.dropout(output, p=p, training=training, inplace=inplace)
#     return output

def test_dropout_sigmoid_linear():
    results = {}
    
    # Test case 1: Basic test with bias, training=True, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_1"] = dropout_sigmoid_linear(input, weight, bias)
    
    # Test case 2: No bias, training=True, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    results["test_case_2"] = dropout_sigmoid_linear(input, weight)
    
    # Test case 3: With bias, training=False, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_3"] = dropout_sigmoid_linear(input, weight, bias, training=False)
    
    # Test case 4: With bias, training=True, inplace=True
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_4"] = dropout_sigmoid_linear(input, weight, bias, inplace=True)
    
    return results

test_results = test_dropout_sigmoid_linear()
