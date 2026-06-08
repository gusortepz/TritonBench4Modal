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


def cholesky_solve(B: Tensor, L: Tensor, upper: bool = False, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the solution of a system of linear equations with complex Hermitian
    or real symmetric positive-definite lhs given its Cholesky decomposition.

    Args:
        B: right-hand side tensor of shape (*, n, k)
        L: tensor of shape (*, n, n) consisting of lower or upper triangular
           Cholesky decompositions of symmetric or Hermitian positive-definite matrices
        upper: flag that indicates whether L is lower triangular or upper triangular.
               Default: False
        out: output tensor. Ignored if None. Default: None

    Returns:
        Solution tensor of shape (*, n, k)
    """
    y = torch.cholesky_solve(B, L, upper=upper)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_cholesky_solve():
    results = {}

    # Test case 1: Lower triangular matrix
    B1 = torch.tensor([[1.0], [2.0]], device='cuda')
    L1 = torch.tensor([[2.0, 0.0], [1.0, 1.0]], device='cuda')
    results["test_case_1"] = cholesky_solve(B1, L1)

    # Test case 2: Upper triangular matrix
    B2 = torch.tensor([[1.0], [2.0]], device='cuda')
    L2 = torch.tensor([[2.0, 1.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = cholesky_solve(B2, L2, upper=True)

    # Test case 3: Batch of matrices, lower triangular
    B3 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L3 = torch.tensor([[[2.0, 0.0], [1.0, 1.0]], [[3.0, 0.0], [1.0, 2.0]]], device='cuda')
    results["test_case_3"] = cholesky_solve(B3, L3)

    # Test case 4: Batch of matrices, upper triangular
    B4 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L4 = torch.tensor([[[2.0, 1.0], [0.0, 1.0]], [[3.0, 1.0], [0.0, 2.0]]], device='cuda')
    results["test_case_4"] = cholesky_solve(B4, L4, upper=True)

    return results

test_results = test_cholesky_solve()
