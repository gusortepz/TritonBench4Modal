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


def softmax_mul(input, other, dim, dtype=None, out=None) -> Tensor:
    """
    Applies the softmax function to the input tensor along the specified dimension,
    and then multiplies the softmaxed values by other.

    Args:
        input (Tensor): The input tensor to apply softmax on.
        other (Tensor or Number): The tensor or number to multiply with the softmaxed values.
        dim (int): The dimension along which softmax will be computed.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
        out (Tensor, optional): The output tensor.
    """
    # Cast input if dtype is specified
    if dtype is not None:
        x = input.to(dtype)
    else:
        x = input

    # Apply softmax along the specified dimension
    sm = F.softmax(x, dim=dim)

    # Multiply by other
    if isinstance(other, Tensor):
        result = sm * other
    else:
        result = sm * other

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax_mul(input, other, dim, dtype=None, out=None):
#     softmaxed = F.softmax(input, dim=dim, dtype=dtype)
#     if isinstance(other, torch.Tensor):
#         result = softmaxed * other
#     else:
#         result = softmaxed * other
#     if out is not None:
#         out.copy_(result)
#         return out
#     return result

def test_softmax_mul():
    results = {}
    
    # Test case 1: Basic test with two tensors
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_1"] = softmax_mul(input1, other1, dim=1)
    
    # Test case 2: Test with scalar multiplication
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other2 = 0.5
    results["test_case_2"] = softmax_mul(input2, other2, dim=1)
    
    # Test case 3: Test with different dtype
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_3"] = softmax_mul(input3, other3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with out parameter
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    out4 = torch.empty_like(input4)
    results["test_case_4"] = softmax_mul(input4, other4, dim=1, out=out4)
    
    return results

test_results = test_softmax_mul()
