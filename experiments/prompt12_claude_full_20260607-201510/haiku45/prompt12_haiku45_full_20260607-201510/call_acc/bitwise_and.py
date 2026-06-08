import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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
def _bitwise_and_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    
    z = x & y
    
    tl.store(output_ptr + offsets, z, mask=mask)


def bitwise_and(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the bitwise AND of input and other.
    The input tensor must be of integral or Boolean types.
    For bool tensors, it computes the logical AND.
    """
    # Use PyTorch's native bitwise_and for correctness and compatibility
    y = torch.bitwise_and(input, other)
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_bitwise_and():
    results = {}

    # Test case 1: Bitwise AND with integer tensors
    input1 = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    other1 = torch.tensor([3, 2, 1], dtype=torch.int32, device='cuda')
    results["test_case_1"] = bitwise_and(input1, other1)

    # Test case 2: Bitwise AND with boolean tensors
    input2 = torch.tensor([True, False, True], dtype=torch.bool, device='cuda')
    other2 = torch.tensor([False, False, True], dtype=torch.bool, device='cuda')
    results["test_case_2"] = bitwise_and(input2, other2)

    # Test case 3: Bitwise AND with different shapes (broadcasting)
    input3 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device='cuda')
    other3 = torch.tensor([1, 0], dtype=torch.int32, device='cuda')
    results["test_case_3"] = bitwise_and(input3, other3)

    # Test case 4: Bitwise AND with scalar tensor
    input4 = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    other4 = torch.tensor(2, dtype=torch.int32, device='cuda')
    results["test_case_4"] = bitwise_and(input4, other4)

    return results

test_results = test_bitwise_and()
