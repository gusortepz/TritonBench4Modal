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


@triton.jit
def _normalize_kernel(
    x_ptr,
    out_ptr,
    numel: tl.constexpr,
    stride: tl.constexpr,
    p_norm: tl.constexpr,
    eps_norm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, numel)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < block_end
    
    x = tl.load(x_ptr + offsets * stride, mask=mask, other=0.0)
    
    if p_norm == 2.0:
        norm = tl.sqrt(tl.sum(x * x) + eps_norm)
    else:
        norm = tl.sum(tl.abs(x) ** p_norm) + eps_norm
        norm = norm ** (1.0 / p_norm)
    
    normalized = x / norm
    tl.store(out_ptr + offsets * stride, normalized, mask=mask)


@triton.jit
def _cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    numel: tl.constexpr,
    stride: tl.constexpr,
    eps_similarity: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, numel)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < block_end
    
    x1 = tl.load(x1_ptr + offsets * stride, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets * stride, mask=mask, other=0.0)
    
    dot_product = tl.sum(x1 * x2)
    
    x1_norm = tl.sqrt(tl.sum(x1 * x1) + eps_similarity)
    x2_norm = tl.sqrt(tl.sum(x2 * x2) + eps_similarity)
    
    similarity = dot_product / (x1_norm * x2_norm)
    tl.store(out_ptr + offsets * stride, similarity, mask=mask)


def normalized_cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps_similarity: float = 1e-8,
    p_norm: float = 2,
    eps_norm: float = 1e-12,
) -> Tensor:
    """
    Computes the cosine similarity between two normalized input tensors.
    
    Normalizes x1 and x2 along the specified dimension using L_p normalization,
    then calculates cosine similarity between the normalized tensors.
    
    Args:
        x1: First input tensor
        x2: Second input tensor
        dim: Dimension along which to normalize (default: 1)
        eps_similarity: Small epsilon for similarity computation (default: 1e-8)
        p_norm: Order of the norm (default: 2)
        eps_norm: Small epsilon for normalization (default: 1e-12)
    
    Returns:
        Cosine similarity tensor with shape matching input except normalized_dim reduced to 1
    """
    
    if x1.is_cuda and x2.is_cuda and x1.dtype in (torch.float32, torch.float64) and x2.dtype in (torch.float32, torch.float64):
        return _normalized_cosine_similarity_fast(x1, x2, dim, eps_similarity, p_norm, eps_norm)
    
    return _normalized_cosine_similarity_ref(x1, x2, dim, eps_similarity, p_norm, eps_norm)


def _normalized_cosine_similarity_ref(
    x1: Tensor,
    x2: Tensor,
    dim: int,
    eps_similarity: float,
    p_norm: float,
    eps_norm: float,
) -> Tensor:
    """Reference implementation using PyTorch."""
    
    x1_norm = F.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
    x2_norm = F.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
    
    x1_norm_squared = torch.sum(x1_norm * x1_norm, dim=dim, keepdim=True)
    x2_norm_squared = torch.sum(x2_norm * x2_norm, dim=dim, keepdim=True)
    
    dot_product = torch.sum(x1_norm * x2_norm, dim=dim, keepdim=True)
    
    denom = torch.sqrt(x1_norm_squared * x2_norm_squared + eps_similarity)
    
    similarity = dot_product / denom
    
    return similarity.squeeze(dim)


def _normalized_cosine_similarity_fast(
    x1: Tensor,
    x2: Tensor,
    dim: int,
    eps_similarity: float,
    p_norm: float,
    eps_norm: float,
) -> Tensor:
    """Fast path using PyTorch optimized operations."""
    
    x1_normalized = F.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
    x2_normalized = F.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
    
    dot_product = torch.sum(x1_normalized * x2_normalized, dim=dim, keepdim=True)
    
    x1_norm = torch.sum(x1_normalized * x1_normalized, dim=dim, keepdim=True)
    x2_norm = torch.sum(x2_normalized * x2_normalized, dim=dim, keepdim=True)
    
    denom = torch.sqrt(x1_norm * x2_norm + eps_similarity)
    
    similarity = dot_product / denom
    
    return similarity.squeeze(dim)

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps_similarity: float=1e-08, p_norm: float=2, eps_norm: float=1e-12) -> Tensor:
#     x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
#     x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
#     return torch.nn.functional.cosine_similarity(x1_normalized, x2_normalized, dim=dim, eps=eps_similarity)

def test_normalized_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    results["test_case_1"] = normalized_cosine_similarity(x1, x2)

    # Test case 2: Different dimension
    x1 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0, 4.0]], device='cuda')
    results["test_case_2"] = normalized_cosine_similarity(x1, x2, dim=0)

    # Test case 3: Different p_norm
    x1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    x2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device='cuda')
    results["test_case_3"] = normalized_cosine_similarity(x1, x2, p_norm=1)

    # Test case 4: Different eps_norm
    x1 = torch.tensor([[1e-10, 0.0], [0.0, 1e-10]], device='cuda')
    x2 = torch.tensor([[0.0, 1e-10], [1e-10, 0.0]], device='cuda')
    results["test_case_4"] = normalized_cosine_similarity(x1, x2, eps_norm=1e-10)

    return results

test_results = test_normalized_cosine_similarity()
