import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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


def normalize_pairwise_distance(
    x1: Tensor,
    x2: Tensor,
    p_distance: float = 2.0,
    eps_distance: float = 1e-6,
    keepdim: bool = False,
    p_norm: float = 2,
    dim_norm: int = 1,
    eps_norm: float = 1e-12,
) -> Tensor:
    """
    Computes the pairwise distance between x1 and x2 using the specified norm,
    then normalizes the resulting distances along the specified dimension.
    
    Args:
        x1: The first input tensor
        x2: The second input tensor, must have the same shape as x1
        p_distance: The norm degree for computing the pairwise distance. Default: 2.0
        eps_distance: Small value to avoid division by zero in pairwise distance calculation. Default: 1e-6
        keepdim: Whether to keep the reduced dimensions in the output. Default: False
        p_norm: The exponent value in the norm formulation for normalization. Default: 2
        dim_norm: The dimension along which normalization is applied. Default: 1
        eps_norm: Small value to avoid division by zero in normalization. Default: 1e-12
    
    Returns:
        Normalized pairwise distances
    """
    
    # Compute pairwise distance using torch.pairwise_distance
    # This computes L_p distance between x1 and x2
    dist = torch.pairwise_distance(x1, x2, p=p_distance, eps=eps_distance, keepdim=True)
    
    # Compute the normalization factor along dim_norm
    # We need to compute the norm of the distances along the specified dimension
    if p_norm == 2:
        # L2 norm: sqrt(sum(x^2))
        norm_factor = torch.norm(dist, p=2, dim=dim_norm, keepdim=True, out=None)
    elif p_norm == 1:
        # L1 norm: sum(|x|)
        norm_factor = torch.norm(dist, p=1, dim=dim_norm, keepdim=True, out=None)
    else:
        # General Lp norm
        norm_factor = torch.norm(dist, p=p_norm, dim=dim_norm, keepdim=True, out=None)
    
    # Add eps_norm to avoid division by zero
    norm_factor = torch.clamp(norm_factor, min=eps_norm)
    
    # Normalize the distances
    normalized_dist = dist / norm_factor
    
    # Remove the keepdim dimension if keepdim is False
    if not keepdim:
        normalized_dist = normalized_dist.squeeze(dim=dim_norm)
    
    return normalized_dist

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
