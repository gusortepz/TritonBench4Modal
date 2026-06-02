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

def _fused_pairwise_distance_normalize_impl(x1, x2, p_norm, eps_norm, eps_distance, keepdim):
    x1_norm = x1 / torch.norm(x1, p=p_norm, dim=-1, keepdim=True).clamp(min=eps_norm)
    x2_norm = x2 / torch.norm(x2, p=p_norm, dim=-1, keepdim=True).clamp(min=eps_norm)
    dist = torch.cdist(x1_norm, x2_norm, p=p_norm)
    dist = dist + eps_distance
    if not keepdim:
        return dist
    return dist.unsqueeze(-1)

try:
    _fused_pairwise_distance_normalize_fast = torch.compile(_fused_pairwise_distance_normalize_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_pairwise_distance_normalize_fast = _fused_pairwise_distance_normalize_impl

def fused_pairwise_distance_normalize(x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    y = _fused_pairwise_distance_normalize_fast(x1, x2, p_norm, eps_norm, eps_distance, keepdim)
    return y

##################################################################################################################################################



import torch

def test_fused_pairwise_distance_normalize():
    results = {}

    # Test case 1: Basic functionality with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = fused_pairwise_distance_normalize(x1, x2)

    # Test case 2: Different p_norm value
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = fused_pairwise_distance_normalize(x1, x2, p_norm=1.0)

    # Test case 3: Different eps_norm value
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = fused_pairwise_distance_normalize(x1, x2, eps_norm=1e-10)

    # Test case 4: keepdim=True
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_4"] = fused_pairwise_distance_normalize(x1, x2, keepdim=True)

    return results

test_results = test_fused_pairwise_distance_normalize()
