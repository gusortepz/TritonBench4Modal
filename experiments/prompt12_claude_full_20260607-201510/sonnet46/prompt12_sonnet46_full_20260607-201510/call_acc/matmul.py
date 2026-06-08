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


def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Matrix product of two tensors.
    
    - 1D tensors: dot product
    - 2D tensors: matrix-matrix product
    - 1D and 2D mixed: matrix-vector product
    - N-D tensors (N > 2): batched matmul with broadcasting
    """
    # Use torch.matmul which handles all dimensionality cases
    if out is not None:
        # For 1D dot product case, torch.matmul with out= is not supported
        # Check if both are 1D (dot product case)
        if input.dim() == 1 and other.dim() == 1:
            y = torch.matmul(input, other)
            out.copy_(y)
            return out
        else:
            return torch.matmul(input, other, out=out)
    else:
        return torch.matmul(input, other)

##################################################################################################################################################



import torch

def test_matmul():
    results = {}

    # Test case 1: Multiplying two 2D tensors
    tensor1 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[5, 6], [7, 8]], device='cuda', dtype=torch.float)
    results["test_case_1"] = matmul(tensor1, tensor2)

    # Test case 2: Multiplying a 1D tensor with a 2D tensor
    tensor1 = torch.tensor([1, 2], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[3, 4], [5, 6]], device='cuda', dtype=torch.float)
    results["test_case_2"] = matmul(tensor1, tensor2)

    # Test case 3: Multiplying a 2D tensor with a 1D tensor
    tensor1 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([5, 6], device='cuda', dtype=torch.float)
    results["test_case_3"] = matmul(tensor1, tensor2)

    # Test case 4: Multiplying two 3D tensors
    tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], device='cuda', dtype=torch.float)
    results["test_case_4"] = matmul(tensor1, tensor2)

    return results

test_results = test_matmul()
