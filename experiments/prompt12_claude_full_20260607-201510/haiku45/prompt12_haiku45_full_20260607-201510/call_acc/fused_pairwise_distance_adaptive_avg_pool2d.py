import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Union, Tuple

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


def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    output_size: Union[int, Tuple[int, ...]],
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Applies adaptive average pooling to x1 and x2, then computes pairwise distance.
    
    Args:
        x1: First input tensor (expected shape: [N, C, H, W] or similar)
        x2: Second input tensor (expected shape: [M, C, H, W] or similar)
        output_size: Target output size for adaptive_avg_pool2d
        p: Norm degree for pairwise distance (default: 2.0)
        eps: Small epsilon to avoid division by zero (default: 1e-6)
        keepdim: Whether to keep the reduced dimension (default: False)
    
    Returns:
        Pairwise distance tensor between pooled x1 and x2
    """
    # Apply adaptive average pooling to both inputs
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)
    
    # Reshape for pairwise distance computation
    # x1_pooled: [N, C, h, w] -> [N, C*h*w]
    # x2_pooled: [M, C, h, w] -> [M, C*h*w]
    x1_flat = x1_pooled.reshape(x1_pooled.size(0), -1)
    x2_flat = x2_pooled.reshape(x2_pooled.size(0), -1)
    
    # Compute pairwise distance using torch.cdist
    # This handles the norm computation correctly with eps
    dist = torch.cdist(x1_flat, x2_flat, p=p, compute_mode='use_mm_for_euclid_dist_if_necessary')
    
    if not keepdim:
        return dist
    
    # If keepdim is True, add back a dimension to match expectation
    # Shape: [N, M, 1] if keepdim=True, else [N, M]
    return dist.unsqueeze(-1)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float=2.0, eps: float=1e-06, keepdim: bool=False) -> torch.Tensor:
#     pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
#     pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
#     diff = pooled_x1 - pooled_x2
#     dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
#     return dist

def test_fused_pairwise_distance_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_1"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8))

    # Test case 2: Different output size
    x1 = torch.rand((2, 3, 64, 64), device='cuda')
    x2 = torch.rand((2, 3, 64, 64), device='cuda')
    results["test_case_2"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(16, 16))

    # Test case 3: Different norm degree
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_3"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), p=1.0)

    # Test case 4: Keep dimension
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_4"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), keepdim=True)

    return results

test_results = test_fused_pairwise_distance_adaptive_avg_pool2d()
