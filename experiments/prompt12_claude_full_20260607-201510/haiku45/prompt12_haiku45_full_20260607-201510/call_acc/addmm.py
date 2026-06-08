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


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: float = 1,
    alpha: float = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs addmm: out = beta * input + alpha * (mat1 @ mat2)
    
    Args:
        input: matrix to be added, broadcastable with (n x p)
        mat1: (n x m) tensor
        mat2: (m x p) tensor
        beta: multiplier for input (default: 1)
        alpha: multiplier for mat1 @ mat2 (default: 1)
        out: optional output tensor
    
    Returns:
        output tensor of shape (n x p)
    """
    # Compute mat1 @ mat2
    matmul_result = torch.matmul(mat1, mat2)
    
    # Handle beta == 0 case: ignore input entirely
    if beta == 0:
        y = alpha * matmul_result
    else:
        # Broadcast input if necessary and apply beta
        y = beta * input + alpha * matmul_result
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch

def test_addmm():
    results = {}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    return results

test_results = test_addmm()
