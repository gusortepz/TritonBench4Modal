import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union, NamedTuple
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


class LDLResult(NamedTuple):
    """Named tuple for LDL factorization result."""
    LD: Tensor
    pivots: Tensor


def ldl_factor(
    A: Tensor,
    *,
    hermitian: bool = False,
    out: Optional[Tuple[Tensor, Tensor]] = None,
) -> LDLResult:
    """
    Computes a compact representation of the LDL factorization of a Hermitian 
    or symmetric (possibly indefinite) matrix.
    
    Supports input of float, double, cfloat and cdouble dtypes. Also supports 
    batches of matrices, and if A is a batch of matrices then the output has 
    the same batch dimensions. When A is complex valued it can be Hermitian 
    (hermitian=True) or symmetric (hermitian=False). The factorization is of 
    the form A = L D L^T. If hermitian is True then transpose operation is 
    the conjugate transpose. L (or U) and D are stored in compact form in LD. 
    They follow the format specified by LAPACK's sytrf function. These tensors 
    may be used in torch.linalg.ldl_solve to solve linear systems.
    
    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch 
                    dimensions consisting of symmetric or Hermitian matrices.
        hermitian (bool, optional): whether to consider the input to be Hermitian 
                                   or symmetric. For real-valued matrices, this 
                                   switch has no effect. Default: `False`.
        out (tuple, optional): tuple of two tensors to write the output to. 
                              Ignored if `None`. Default: `None`.
    
    Returns:
        A named tuple `(LD, pivots)` containing:
            - LD (Tensor): the compact representation of L and D
            - pivots (Tensor): the pivot indices
    """
    # Use torch.linalg.ldl_factor directly
    LD, pivots = torch.linalg.ldl_factor(A, hermitian=hermitian)
    
    # Handle out parameter
    if out is not None:
        out_LD, out_pivots = out
        out_LD.copy_(LD)
        out_pivots.copy_(pivots)
        return LDLResult(out_LD, out_pivots)
    
    return LDLResult(LD, pivots)

##################################################################################################################################################



import torch

def test_ldl_factor():
    results = {}

    # Test case 1: Symmetric matrix
    A1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device='cuda')
    results["test_case_1"] = ldl_factor(A1)

    # Test case 2: Hermitian matrix
    A2 = torch.tensor([[2.0, 1.0j], [-1.0j, 2.0]], device='cuda')
    results["test_case_2"] = ldl_factor(A2, hermitian=True)

    # Test case 3: Batch of symmetric matrices
    A3 = torch.tensor([[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.5], [0.5, 2.0]]], device='cuda')
    results["test_case_3"] = ldl_factor(A3)

    # Test case 4: Batch of Hermitian matrices
    A4 = torch.tensor([[[2.0, 1.0j], [-1.0j, 2.0]], [[3.0, 0.5j], [-0.5j, 3.0]]], device='cuda')
    results["test_case_4"] = ldl_factor(A4, hermitian=True)

    return results

test_results = test_ldl_factor()
