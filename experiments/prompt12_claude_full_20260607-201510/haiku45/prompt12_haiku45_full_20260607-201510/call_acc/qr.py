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


def qr(A: Tensor, mode: str = 'reduced', *, out: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the QR decomposition of a matrix.
    
    Supports input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices, and if A is a batch of matrices then 
    the output has the same batch dimensions.
    
    Args:
        A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        mode (str, optional): one of `'reduced'`, `'complete'`, `'r'`. 
            Controls the shape of the returned tensors. Default: `'reduced'`.
        out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.
    
    Returns:
        (Q, R) where Q and R are tensors satisfying A = Q @ R.
        - In 'reduced' mode: Q is (*, m, k) and R is (*, k, n) where k = min(m, n)
        - In 'complete' mode: Q is (*, m, m) and R is (*, m, n)
        - In 'r' mode: only R is returned as (*, k, n) where k = min(m, n)
    """
    
    # Validate mode parameter
    if mode not in ('reduced', 'complete', 'r'):
        raise ValueError(f"mode must be one of 'reduced', 'complete', 'r', got {mode}")
    
    # Use torch.linalg.qr which handles all dtypes, batches, and modes correctly
    if mode == 'r':
        # For 'r' mode, torch.linalg.qr still returns (Q, R), we just return R
        Q, R = torch.linalg.qr(A, mode='reduced')
        if out is not None:
            # In 'r' mode with out tuple, store R in the second position
            # This is semantically consistent with returning only R
            if len(out) >= 1 and out[0] is not None:
                out[0].copy_(R)
            if len(out) >= 2 and out[1] is not None:
                out[1].copy_(R)
            return out
        return R
    else:
        # For 'reduced' and 'complete' modes, return both Q and R
        Q, R = torch.linalg.qr(A, mode=mode)
        
        if out is not None:
            if len(out) >= 1 and out[0] is not None:
                out[0].copy_(Q)
            if len(out) >= 2 and out[1] is not None:
                out[1].copy_(R)
            return tuple(out[:2])
        
        return Q, R

##################################################################################################################################################



import torch

def test_qr():
    results = {}

    # Test case 1: reduced mode, 2x2 matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    Q1, R1 = qr(A1, mode='reduced')
    results["test_case_1"] = (Q1.cpu(), R1.cpu())

    # Test case 2: complete mode, 3x2 matrix
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    Q2, R2 = qr(A2, mode='complete')
    results["test_case_2"] = (Q2.cpu(), R2.cpu())

    # Test case 3: r mode, 2x3 matrix
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    Q3, R3 = qr(A3, mode='r')
    results["test_case_3"] = (Q3.cpu(), R3.cpu())

    # Test case 4: reduced mode, batch of 2x2 matrices
    A4 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    Q4, R4 = qr(A4, mode='reduced')
    results["test_case_4"] = (Q4.cpu(), R4.cpu())

    return results

test_results = test_qr()
