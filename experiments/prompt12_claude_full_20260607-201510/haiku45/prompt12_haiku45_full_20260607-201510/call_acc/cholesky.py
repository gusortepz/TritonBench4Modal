import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl
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


def cholesky(A: Tensor, *, upper: bool = False, out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the Cholesky decomposition of a complex Hermitian or real symmetric
    positive-definite matrix.
    
    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of symmetric or Hermitian positive-definite matrices.
    
    Keyword args:
        upper (bool, optional): whether to return an upper triangular matrix.
            The tensor returned with upper=True is the conjugate transpose of the tensor
            returned with upper=False. Default: False.
        out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
    
    Returns:
        Tensor: the Cholesky decomposition of A.
    """
    # Use torch.linalg.cholesky as the reference implementation
    # It handles all dtypes (float, double, cfloat, cdouble) and batch dimensions correctly
    L = torch.linalg.cholesky(A, upper=upper)
    
    if out is not None:
        out.copy_(L)
        return out
    return L

##################################################################################################################################################



import torch

def test_cholesky():
    results = {}
    
    # Test case 1: Real symmetric positive-definite matrix, lower triangular
    A1 = torch.randn(2, 2, device='cuda', dtype=torch.float64)
    A1 = A1 @ A1.T + torch.eye(2, device='cuda', dtype=torch.float64)
    L1 = cholesky(A1)
    results["test_case_1"] = L1
    
    # Test case 2: Real symmetric positive-definite matrix, upper triangular
    A2 = torch.randn(2, 2, device='cuda', dtype=torch.float64)
    A2 = A2 @ A2.T + torch.eye(2, device='cuda', dtype=torch.float64)
    L2 = cholesky(A2, upper=True)
    results["test_case_2"] = L2
    
    # Test case 3: Complex Hermitian positive-definite matrix, lower triangular
    A3 = torch.randn(2, 2, device='cuda', dtype=torch.complex128)
    A3 = A3 @ A3.T.conj() + torch.eye(2, device='cuda', dtype=torch.complex128)
    L3 = cholesky(A3)
    results["test_case_3"] = L3
    
    # Test case 4: Complex Hermitian positive-definite matrix, upper triangular
    A4 = torch.randn(2, 2, device='cuda', dtype=torch.complex128)
    A4 = A4 @ A4.T.conj() + torch.eye(2, device='cuda', dtype=torch.complex128)
    L4 = cholesky(A4, upper=True)
    results["test_case_4"] = L4
    
    return results

test_results = test_cholesky()
