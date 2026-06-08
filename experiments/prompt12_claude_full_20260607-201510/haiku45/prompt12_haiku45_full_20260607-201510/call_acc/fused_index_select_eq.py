import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Union
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


def fused_index_select_eq(
    input: Tensor,
    dim: int,
    index: Tensor,
    other: Union[Tensor, float],
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs a fused operation combining index selection and element-wise equality comparison.
    Selects elements from the input tensor along a specified dimension using provided indices
    and then performs an element-wise equality comparison between the selected elements and
    another tensor or scalar.
    
    Args:
        input (Tensor): The input tensor X.
        dim (int): The dimension along which to index.
        index (IntTensor or LongTensor): The indices to select along dimension dim.
        other (Tensor or float): The tensor or value Y to compare with the selected tensor.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None
    
    Returns:
        Tensor: A boolean tensor of the same shape as the selected elements, indicating
                where the comparisons are true.
    """
    
    selected = torch.index_select(input, dim, index)
    
    if isinstance(other, Tensor):
        result = selected == other
    else:
        result = selected == other
    
    if out is not None:
        out.copy_(result)
        return out
    
    return result

##################################################################################################################################################



import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()
