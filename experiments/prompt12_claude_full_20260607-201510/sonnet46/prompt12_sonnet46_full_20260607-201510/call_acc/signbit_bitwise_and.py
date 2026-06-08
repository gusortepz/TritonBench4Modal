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


def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and the bitwise AND operation on the input tensors.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types.
    
    Returns:
        Tuple[Tensor, Tensor]: (signbit_result, bitwise_and_result)
    """
    signbit_result = torch.signbit(input)
    bitwise_and_result = torch.bitwise_and(other, other)  # placeholder
    # Actually compute bitwise AND between input (cast if needed) and other
    # But per the description, bitwise_and is between input and other
    # input might be float, so we need to handle this carefully
    # Looking at the example: a is float, b is int8, and result is int8 zeros
    # torch.bitwise_and requires integral or boolean types
    # If input is float, we need to cast or handle appropriately
    # The example shows bitwise_and_result = tensor([0, 0, 0, 0], dtype=torch.int8)
    # which suggests b & b or similar... actually 1&1=1, 0&0=0, 1&1=1, 1&1=1 != [0,0,0,0]
    # Wait, let me re-read: input=a (float), other=b (int8)
    # torch.bitwise_and(a, b) would fail since a is float
    # The example result is [0,0,0,0] which matches b & 0 or something
    # Actually looking more carefully: perhaps it's bitwise_and(other, other) 
    # but that gives [1,0,1,1] not [0,0,0,0]
    # The only way to get [0,0,0,0] is if we cast input to int8 first
    # float -> int8: 0.7->0, -1.2->-1, 0.->0, 2.3->2
    # int8: 0, -1 (=0xFF), 0, 2 = [0x00, 0xFF, 0x00, 0x02]
    # other: [1, 0, 1, 1]
    # AND: [0&1, 0xFF&0, 0&1, 2&1] = [0, 0, 0, 0] ✓
    # So bitwise_and is between input.to(other.dtype) and other
    
    if input.is_floating_point() or input.is_complex():
        input_int = input.to(other.dtype)
    else:
        input_int = input.to(other.dtype)
    
    bitwise_and_result = torch.bitwise_and(input_int, other)
    
    return signbit_result, bitwise_and_result

##################################################################################################################################################



import torch
from typing import Tuple

# def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     signbit_result = torch.signbit(input)
#     bitwise_and_result = input.to(torch.int8) & other.to(torch.int8)
#     return (signbit_result, bitwise_and_result)

def test_signbit_bitwise_and():
    results = {}

    # Test case 1: Positive and negative floats with integer tensor
    a = torch.tensor([0.7, -1.2, 0., 2.3], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_1"] = signbit_bitwise_and(a, b)

    # Test case 2: All negative floats with integer tensor
    a = torch.tensor([-0.7, -1.2, -0.1, -2.3], device='cuda')
    b = torch.tensor([1, 1, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_2"] = signbit_bitwise_and(a, b)

    # Test case 3: Mixed positive and zero floats with boolean tensor
    a = torch.tensor([0.0, 1.2, 0.0, 2.3], device='cuda')
    b = torch.tensor([True, False, True, True], dtype=torch.bool, device='cuda')
    results["test_case_3"] = signbit_bitwise_and(a, b)

    # Test case 4: All zero floats with integer tensor
    a = torch.tensor([0.0, 0.0, 0.0, 0.0], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    results["test_case_4"] = signbit_bitwise_and(a, b)

    return results

test_results = test_signbit_bitwise_and()
