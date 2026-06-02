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

def _matrix_vector_dot_impl(A, x, y, alpha, beta):
    y_new = alpha * torch.mv(A, x) + beta * y
    y.copy_(y_new)
    return torch.dot(y_new, x)

try:
    _matrix_vector_dot_fast = torch.compile(_matrix_vector_dot_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _matrix_vector_dot_fast = _matrix_vector_dot_impl

def matrix_vector_dot(A: Tensor, x: Tensor, y: Tensor, alpha: float, beta: float) -> Tensor:
    return _matrix_vector_dot_fast(A, x, y, alpha, beta)

##################################################################################################################################################



import torch
from torch import Tensor

def test_matrix_vector_dot():
    results = {}
    
    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([0.0, 0.0], device='cuda')
    alpha = 1.0
    beta = 0.0
    results["test_case_1"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_2"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 3
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([2.0, 3.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 0.5
    beta = 0.5
    results["test_case_3"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 4
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([2.0, 2.0], device='cuda')
    alpha = 2.0
    beta = 0.5
    results["test_case_4"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    return results

test_results = test_matrix_vector_dot()
