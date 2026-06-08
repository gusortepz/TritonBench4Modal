import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple
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


def svd(
    A: Tensor,
    full_matrices: bool = True,
    *,
    driver: Optional[str] = None,
    out: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes the singular value decomposition (SVD) of a matrix.
    
    Args:
        A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        full_matrices (bool, optional): controls whether to compute the full or reduced SVD.
            Default: `True`.
    
    Keyword args:
        driver (str, optional): name of the cuSOLVER method to be used. 
            Only works on CUDA inputs. Options: `None`, `gesvd`, `gesvdj`, `gesvda`.
            Default: `None`.
        out (tuple, optional): output tuple of three tensors (U, S, Vh).
            Ignored if `None`.
    
    Returns:
        Tuple[Tensor, Tensor, Tensor]: (U, S, Vh) where U and Vh are unitary matrices
            and S contains the singular values in descending order.
    """
    
    # Use torch.linalg.svd as the reference implementation
    # This handles all complex logic: batching, dtype support, full/reduced SVD, driver selection
    if driver is not None:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices, driver=driver)
    else:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Handle out parameter if provided
    if out is not None:
        out_U, out_S, out_Vh = out
        out_U.copy_(U)
        out_S.copy_(S)
        out_Vh.copy_(Vh)
        return out_U, out_S, out_Vh
    
    return U, S, Vh

##################################################################################################################################################



import torch

def test_svd():
    results = {}

    # Test case 1: 2x2 matrix, full_matrices=True
    A1 = torch.tensor([[3.0, 1.0], [1.0, 3.0]], device='cuda')
    U1, S1, Vh1 = svd(A1, full_matrices=True)
    results["test_case_1"] = (U1.cpu(), S1.cpu(), Vh1.cpu())

    # Test case 2: 3x2 matrix, full_matrices=False
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    U2, S2, Vh2 = svd(A2, full_matrices=False)
    results["test_case_2"] = (U2.cpu(), S2.cpu(), Vh2.cpu())

    # Test case 3: 2x3 matrix, full_matrices=True
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    U3, S3, Vh3 = svd(A3, full_matrices=True)
    results["test_case_3"] = (U3.cpu(), S3.cpu(), Vh3.cpu())

    # Test case 4: 3x3 matrix, full_matrices=False
    A4 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
    U4, S4, Vh4 = svd(A4, full_matrices=False)
    results["test_case_4"] = (U4.cpu(), S4.cpu(), Vh4.cpu())

    return results

test_results = test_svd()
