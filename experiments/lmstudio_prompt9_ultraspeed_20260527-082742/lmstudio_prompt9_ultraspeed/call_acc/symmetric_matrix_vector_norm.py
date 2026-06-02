import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def _sym_mv_norm_impl(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float) -> torch.Tensor:
    y = alpha * torch.mv(A, x) + beta
    return torch.norm(y, p=p)

try:
    _sym_mv_norm_fast = torch.compile(_sym_mv_norm_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _sym_mv_norm_fast = _sym_mv_norm_impl

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    return _sym_mv_norm_fast(A, x, alpha, beta, p)

##################################################################################################################################################



import torch

def test_symmetric_matrix_vector_norm():
    results = {}

    # Test case 1: Basic test with default p value
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_1"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 2: Different alpha and beta values
    alpha = 2.0
    beta = 0.5
    results["test_case_2"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 3: Different p value (1-norm)
    alpha = 1.0
    beta = 1.0
    p = 1.0
    results["test_case_3"] = symmetric_matrix_vector_norm(A, x, alpha, beta, p).item()

    # Test case 4: Larger matrix and vector
    A = torch.tensor([[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 3.0]], device='cuda')
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha = 1.5
    beta = 0.5
    results["test_case_4"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    return results

test_results = test_symmetric_matrix_vector_norm()
