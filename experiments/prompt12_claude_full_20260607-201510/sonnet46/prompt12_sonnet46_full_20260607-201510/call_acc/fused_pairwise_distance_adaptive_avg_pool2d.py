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


def _fused_impl(x1: Tensor, x2: Tensor, output_size, p: float, eps: float, keepdim: bool) -> Tensor:
    pooled1 = F.adaptive_avg_pool2d(x1, output_size)
    pooled2 = F.adaptive_avg_pool2d(x2, output_size)
    # Flatten to (batch, features) for pairwise distance
    b = pooled1.shape[0]
    flat1 = pooled1.view(b, -1)
    flat2 = pooled2.view(b, -1)
    dist = F.pairwise_distance(flat1, flat2, p=p, eps=eps, keepdim=keepdim)
    return dist


try:
    _fused_fast = torch.compile(_fused_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_fast = _fused_impl


def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    try:
        return _fused_fast(x1, x2, output_size, p, eps, keepdim)
    except Exception:
        return _fused_impl(x1, x2, output_size, p, eps, keepdim)

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
