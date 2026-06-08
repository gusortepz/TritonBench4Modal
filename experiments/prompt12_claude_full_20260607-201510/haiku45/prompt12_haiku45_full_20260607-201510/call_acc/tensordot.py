import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union, List
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


def tensordot(
    a: Tensor,
    b: Tensor,
    dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]
) -> Tensor:
    """
    Returns a contraction of a and b over multiple dimensions.
    Implements a generalized matrix product.
    
    Args:
        a (Tensor): Left tensor to contract
        b (Tensor): Right tensor to contract
        dims (int or Tuple[List[int], List[int]] or List[List[int]]): 
            number of dimensions to contract or explicit lists of dimensions
    
    Returns:
        Tensor: Contracted result
    """
    return torch.tensordot(a, b, dims=dims)

##################################################################################################################################################



import torch
from typing import Union, List, Tuple

def test_tensordot():
    results = {}
    
    # 示例 1
    a = torch.arange(60.).reshape(3, 4, 5)
    b = torch.arange(24.).reshape(4, 3, 2)
    results["test_case_1"] = tensordot(a, b, dims=([1, 0], [0, 1]))

    # 示例 2 (在CUDA设备上)
    a = torch.randn(3, 4, 5, device='cuda')
    b = torch.randn(4, 5, 6, device='cuda')
    results["test_case_2"] = tensordot(a, b, dims=2).cpu()

    # 示例 3 (多维收缩)
    a = torch.randn(3, 5, 4, 6)
    b = torch.randn(6, 4, 5, 3)
    results["test_case_3"] = tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
    
    return results

test_results = test_tensordot()
