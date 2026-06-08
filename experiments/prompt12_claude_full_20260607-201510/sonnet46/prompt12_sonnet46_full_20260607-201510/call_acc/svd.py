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


def svd(
    A: Tensor,
    full_matrices: bool = True,
    *,
    driver: Optional[str] = None,
    out: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    # SVD is a solver/decomposition — direct PyTorch (Route A)
    result = torch.linalg.svd(A, full_matrices=full_matrices, driver=driver)
    U, S, Vh = result.U, result.S, result.Vh

    if out is not None:
        out[0].copy_(U)
        out[1].copy_(S)
        out[2].copy_(Vh)
        return (out[0], out[1], out[2])

    return (U, S, Vh)

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
