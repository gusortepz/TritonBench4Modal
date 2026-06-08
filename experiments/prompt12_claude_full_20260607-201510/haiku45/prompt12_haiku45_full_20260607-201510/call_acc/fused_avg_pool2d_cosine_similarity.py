import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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


def fused_avg_pool2d_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes cosine similarity between x1 and x2 along dim=1,
    adds a singleton dimension via unsqueeze, and applies 2D average pooling.
    
    Args:
        x1: Input tensor of shape (batch, features, ...).
        x2: Input tensor of shape (batch, features, ...).
        kernel_size: Size of the pooling kernel.
        stride: Stride of the pooling. Defaults to kernel_size if None.
        padding: Padding for the pooling operation.
        eps: Small value for numerical stability in cosine similarity.
    
    Returns:
        Tensor after cosine similarity, unsqueeze, and avg_pool2d.
    """
    if stride is None:
        stride = kernel_size
    
    # Compute cosine similarity along dim=1
    cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps)
    
    # Add singleton dimension (unsqueeze to make it 4D for avg_pool2d)
    cos_sim = cos_sim.unsqueeze(1)
    
    # Apply 2D average pooling
    output = F.avg_pool2d(cos_sim, kernel_size=kernel_size, stride=stride, padding=padding)
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, eps: float=1e-08) -> torch.Tensor:
#     cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps)
#     cosine_sim = cosine_sim.unsqueeze(1)
#     if stride is None:
#         stride = kernel_size
#     pooled_result = F.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)
#     return pooled_result

def test_fused_avg_pool2d_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default stride and padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_1"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2)

    # Test case 2: Test with specified stride
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_2"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, stride=1)

    # Test case 3: Test with specified padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_3"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, padding=1)

    # Test case 4: Test with different eps value
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_4"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, eps=1e-6)

    return results

test_results = test_fused_avg_pool2d_cosine_similarity()
