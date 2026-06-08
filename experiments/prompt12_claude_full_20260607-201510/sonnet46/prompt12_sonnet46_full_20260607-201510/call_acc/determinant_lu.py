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


def determinant_lu(A: Tensor, *, pivot: bool = True, out: Optional[Tensor] = None) -> Tensor:
    if pivot:
        LU, pivots = torch.linalg.lu_factor(A)
        # Diagonal of U
        diag_U = torch.diagonal(LU, dim1=-2, dim2=-1)
        # Product of diagonal elements
        det = torch.prod(diag_U, dim=-1)
        # Compute sign from pivots (LAPACK 1-indexed)
        # pivots shape: (*batch, n)
        n = pivots.shape[-1]
        batch_shape = pivots.shape[:-1]
        # Count number of swaps: positions where pivots[i] != i+1
        idx = torch.arange(1, n + 1, device=pivots.device, dtype=pivots.dtype)
        # Broadcast idx to batch shape
        swaps = (pivots != idx).sum(dim=-1)
        # Sign: (-1)^swaps
        sign = torch.where(swaps % 2 == 0,
                           torch.ones(batch_shape, dtype=det.real.dtype if det.is_complex() else det.dtype, device=det.device),
                           -torch.ones(batch_shape, dtype=det.real.dtype if det.is_complex() else det.dtype, device=det.device))
        if det.is_complex():
            sign = sign.to(det.dtype)
        det = det * sign
    else:
        # Without pivoting: just product of diagonal of LU factor (no pivot correction)
        LU, _ = torch.linalg.lu_factor(A, pivot=False)
        diag_U = torch.diagonal(LU, dim1=-2, dim2=-1)
        det = torch.prod(diag_U, dim=-1)

    if out is not None:
        out.copy_(det)
        return out
    return det

##################################################################################################################################################



import torch

def test_determinant_lu():
    results = {}

    # Test case 1: 2x2 matrix with pivot=True
    A1 = torch.tensor([[3.0, 1.0], [2.0, 4.0]], device='cuda')
    results["test_case_1"] = determinant_lu(A1)

    # Test case 2: 3x3 matrix with pivot=False
    A2 = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], device='cuda')
    results["test_case_2"] = determinant_lu(A2, pivot=False)

    # Test case 3: Batch of 2x2 matrices with pivot=True
    A3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = determinant_lu(A3)

    # Test case 4: 4x4 matrix with pivot=True
    A4 = torch.tensor([[1.0, 0.0, 2.0, -1.0],
                       [3.0, 0.0, 0.0, 5.0],
                       [2.0, 1.0, 4.0, -3.0],
                       [1.0, 0.0, 5.0, 0.0]], device='cuda')
    results["test_case_4"] = determinant_lu(A4)

    return results

test_results = test_determinant_lu()
