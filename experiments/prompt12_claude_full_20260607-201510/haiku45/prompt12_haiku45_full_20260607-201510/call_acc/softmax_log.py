import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
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


def softmax_log(input: Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None) -> Tensor:
    """
    Applies natural logarithm element-wise, followed by softmax along the specified dimension.
    
    Args:
        input (Tensor): The input tensor on which logarithm and softmax are applied.
        dim (int): The dimension along which softmax will be computed. Default: -1.
        dtype (torch.dtype, optional): The desired data type of the returned tensor. 
                                       If specified, the input tensor is cast to dtype 
                                       before the operation is performed. Default: None.
    
    Returns:
        Tensor: The result tensor with log and softmax applied.
    """
    # Cast input to dtype if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Apply natural logarithm element-wise
    log_input = torch.log(input)
    
    # Apply softmax along the specified dimension
    result = F.softmax(log_input, dim=dim)
    
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax_log(input, dim=-1, dtype=None):
#     if dtype is not None:
#         input = input.to(dtype)
#     log_input = input.log()
#     return F.softmax(log_input, dim=dim)

def test_softmax_log():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = softmax_log(input_tensor)

    # Test case 2: Specifying a different dimension
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = softmax_log(input_tensor, dim=0)

    # Test case 3: Specifying a different dtype
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = softmax_log(input_tensor, dtype=torch.float64)

    # Test case 4: Larger tensor
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_4"] = softmax_log(input_tensor)

    return results

test_results = test_softmax_log()
