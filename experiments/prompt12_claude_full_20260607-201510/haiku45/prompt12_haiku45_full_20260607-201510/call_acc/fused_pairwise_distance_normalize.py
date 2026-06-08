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


def fused_pairwise_distance_normalize(
    x1: Tensor,
    x2: Tensor,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
    eps_distance: float = 1e-6,
    keepdim: bool = False,
) -> Tensor:
    """
    Computes the pairwise distance between two input tensors x1 and x2
    after normalizing both tensors along the last dimension.
    
    Args:
        x1 (Tensor): First input tensor of shape (..., d).
        x2 (Tensor): Second input tensor of shape (..., d).
        p_norm (float, optional): The exponent value in the norm for normalization. Default: 2.0.
        eps_norm (float, optional): Small value to avoid division by zero during normalization. Default: 1e-12.
        eps_distance (float, optional): Small value to avoid division by zero in distance calculation. Default: 1e-6.
        keepdim (bool, optional): If True, retains the last dimension in the output. Default: False.
    
    Returns:
        Tensor: Pairwise distances between normalized x1 and x2.
    """
    
    # Normalize x1 along the last dimension
    norm_x1 = torch.norm(x1, p=p_norm, dim=-1, keepdim=True)
    norm_x1 = torch.clamp(norm_x1, min=eps_norm)
    x1_normalized = x1 / norm_x1
    
    # Normalize x2 along the last dimension
    norm_x2 = torch.norm(x2, p=p_norm, dim=-1, keepdim=True)
    norm_x2 = torch.clamp(norm_x2, min=eps_norm)
    x2_normalized = x2 / norm_x2
    
    # Compute pairwise distance using L2 norm
    # For two normalized vectors, the L2 distance is sqrt(2 - 2*dot_product)
    # General pairwise distance: ||a - b||_p
    
    if x1_normalized.dim() == 1 and x2_normalized.dim() == 1:
        # Both are 1D: compute single distance
        diff = x1_normalized - x2_normalized
        distance = torch.norm(diff, p=p_norm)
        if keepdim:
            distance = distance.unsqueeze(-1)
    else:
        # General case: compute pairwise distances
        # Reshape for broadcasting if needed
        x1_shape = x1_normalized.shape[:-1]
        x2_shape = x2_normalized.shape[:-1]
        d = x1_normalized.shape[-1]
        
        # Flatten batch dimensions for pairwise computation
        x1_flat = x1_normalized.reshape(-1, d)  # (N, d)
        x2_flat = x2_normalized.reshape(-1, d)  # (M, d)
        
        N = x1_flat.shape[0]
        M = x2_flat.shape[0]
        
        # Compute pairwise differences: (N, M, d)
        x1_exp = x1_flat.unsqueeze(1)  # (N, 1, d)
        x2_exp = x2_flat.unsqueeze(0)  # (1, M, d)
        diff = x1_exp - x2_exp  # (N, M, d)
        
        # Compute norm along last dimension
        distance = torch.norm(diff, p=p_norm, dim=-1)  # (N, M)
        
        # Reshape back to original batch shape
        output_shape = list(x1_shape) + list(x2_shape)
        distance = distance.reshape(output_shape)
        
        if keepdim:
            distance = distance.unsqueeze(-1)
    
    return distance

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
