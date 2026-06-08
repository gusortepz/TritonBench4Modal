import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

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


def mul(input: Tensor, other: Union[Tensor, float, int, complex], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Multiplies the input tensor by another tensor or a number, supporting broadcasting 
    to a common shape, type promotion, and integer, float, and complex inputs.
    
    Args:
        input: the input tensor.
        other: the tensor or number to multiply input by.
        out: optional output tensor.
    
    Returns:
        The result of input * other.
    """
    y = torch.mul(input, other)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_mul():
    results = {}

    # Test case 1: Multiply two tensors with broadcasting
    input1 = torch.tensor([1, 2, 3], device='cuda')
    other1 = torch.tensor([[1], [2], [3]], device='cuda')
    results["test_case_1"] = mul(input1, other1)

    # Test case 2: Multiply tensor by a scalar
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.5
    results["test_case_2"] = mul(input2, other2)

    # Test case 3: Multiply complex tensors
    input3 = torch.tensor([1+2j, 3+4j], device='cuda')
    other3 = torch.tensor([5+6j, 7+8j], device='cuda')
    results["test_case_3"] = mul(input3, other3)

    # Test case 4: Multiply integer tensor by a float tensor
    input4 = torch.tensor([1, 2, 3], device='cuda')
    other4 = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    results["test_case_4"] = mul(input4, other4)

    return results

test_results = test_mul()
