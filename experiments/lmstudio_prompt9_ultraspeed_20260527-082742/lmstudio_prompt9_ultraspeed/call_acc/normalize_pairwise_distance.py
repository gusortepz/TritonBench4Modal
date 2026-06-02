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

def _normalize_pairwise_distance_impl(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm):
    diff = x1 - x2
    dist = torch.linalg.vector_norm(diff, ord=p_distance, dim=-1, keepdim=keepdim) + eps_distance
    return F.normalize(dist, p=p_norm, dim=dim_norm, eps=eps_norm)

try:
    _normalize_pairwise_distance_fast = torch.compile(_normalize_pairwise_distance_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _normalize_pairwise_distance_fast = _normalize_pairwise_distance_impl

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12, *, out=None):
    y = _normalize_pairwise_distance_fast(x1, x2, p_distance, eps_distance, keepdim, p_norm, dim_norm, eps_norm)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-06, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
#     pairwise_distance = torch.norm(x1 - x2, p=p_distance, dim=-1, keepdim=keepdim)
#     pairwise_distance = pairwise_distance + eps_distance
#     normed_distance = pairwise_distance / torch.norm(pairwise_distance, p=p_norm, dim=dim_norm, keepdim=True).clamp(min=eps_norm)
#     return normed_distance

def test_normalize_pairwise_distance():
    results = {}
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[1.0, 2.5], [2.5, 4.0]])
    
    # Compute the normalized pairwise distance
    results["test_case_1"] = normalize_pairwise_distance(x1, x2, p_distance=2.0, dim_norm=0)
    # Normalize along a different dimension
    results["test_case_2"] = normalize_pairwise_distance(x1, x2, p_distance=1.0, dim_norm=0)

    return results

test_results = test_normalize_pairwise_distance()
