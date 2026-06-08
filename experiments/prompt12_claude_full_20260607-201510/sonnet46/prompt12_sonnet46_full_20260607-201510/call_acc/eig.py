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


def eig(A: Tensor, *, out: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the eigenvalue decomposition of a square matrix.

    Supports float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices.

    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch
                    dimensions consisting of diagonalizable matrices.
        out (tuple, optional): output tuple of two tensors. Ignored if `None`.
                               Default: `None`.

    Returns:
        A named tuple (eigenvalues, eigenvectors) where:
            - eigenvalues has shape `(*, n)`
            - eigenvectors has shape `(*, n, n)`
    """
    if out is not None:
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        out[0].resize_as_(eigenvalues).copy_(eigenvalues)
        out[1].resize_as_(eigenvectors).copy_(eigenvectors)
        return (out[0], out[1])
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        return (eigenvalues, eigenvectors)

##################################################################################################################################################



import torch

# def eig(A):
#     (eigenvalues, eigenvectors) = torch.linalg.eig(A)
#     return (eigenvalues, eigenvectors)

def test_eig():
    results = {}

    # Test case 1: 2x2 matrix with distinct eigenvalues
    A1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], device='cuda')
    results["test_case_1"] = eig(A1)

    # Test case 2: 2x2 matrix with repeated eigenvalues
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = eig(A2)

    # Test case 3: 3x3 matrix with complex eigenvalues
    A3 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
    results["test_case_3"] = eig(A3)

    # Test case 4: 3x3 matrix with real eigenvalues
    A4 = torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]], device='cuda')
    results["test_case_4"] = eig(A4)

    return results

test_results = test_eig()
