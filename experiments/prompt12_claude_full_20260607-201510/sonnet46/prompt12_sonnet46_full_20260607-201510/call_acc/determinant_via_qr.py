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


def determinant_via_qr(A: Tensor, *, mode: str = 'reduced', out: Optional[Tensor] = None) -> Tensor:
    """
    Computes the determinant of a square matrix using QR decomposition.
    
    Performs QR decomposition of a square matrix A in K^{n×n} and computes
    the determinant by taking the product of the diagonal elements of R,
    multiplied by the determinant of Q (which is ±1 for orthogonal/unitary Q).
    """
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Product of diagonal elements of R
    diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
    prod_diag_R = torch.prod(diag_R, dim=-1)
    
    # det(Q) for orthogonal/unitary Q - accounts for reflections
    det_Q = torch.linalg.det(Q)
    
    result = det_Q * prod_diag_R
    
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_determinant_via_qr():
    results = {}

    # Test case 1: 2x2 matrix, reduced mode
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_via_qr(A1)

    # Test case 2: 3x3 matrix, reduced mode
    A2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device='cuda')
    results["test_case_2"] = determinant_via_qr(A2)

    # Test case 3: 2x2 matrix, complete mode
    A3 = torch.tensor([[2.0, 3.0], [1.0, 4.0]], device='cuda')
    results["test_case_3"] = determinant_via_qr(A3, mode='complete')

    # Test case 4: 3x3 matrix, complete mode
    A4 = torch.tensor([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [4.0, 1.0, 3.0]], device='cuda')
    results["test_case_4"] = determinant_via_qr(A4, mode='complete')

    return results

test_results = test_determinant_via_qr()
