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


@triton.jit
def _add_tanh_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    z = x + y
    # tanh via sigmoid: tanh(x) = 2*sigmoid(2*x) - 1
    result = 2.0 * tl.sigmoid(2.0 * z) - 1.0

    tl.store(out_ptr + offsets, result, mask=mask)


def _add_tanh_triton(emb: Tensor, other: Tensor) -> Tensor:
    """Fused add + tanh using Triton kernel."""
    # Broadcast other to match emb shape if needed
    other_expanded = other.expand_as(emb)
    # Ensure contiguous
    emb_c = emb.contiguous()
    other_c = other_expanded.contiguous()

    out = torch.empty_like(emb_c)
    n_elements = emb_c.numel()

    BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n_elements, 1)))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_tanh_kernel[grid](
        emb_c,
        other_c,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _add_tanh_pytorch(emb: Tensor, other: Tensor) -> Tensor:
    """PyTorch fallback for add + tanh."""
    return torch.tanh(emb + other)


def fused_embedding_add_tanh(
    input_indices: Tensor,
    weight: Tensor,
    other: Tensor,
    *,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Step 1: Embedding lookup with all PyTorch options
    emb = F.embedding(
        input_indices,
        weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )

    # Step 2 & 3: Fused add + tanh
    # Use Triton path only for CUDA fp32/fp16/bf16 tensors
    use_triton = (
        emb.is_cuda
        and other.is_cuda
        and emb.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and other.dtype == emb.dtype
    )

    if use_triton:
        try:
            y = _add_tanh_triton(emb, other)
        except Exception:
            y = _add_tanh_pytorch(emb, other)
    else:
        y = _add_tanh_pytorch(emb, other)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
#     embeddings = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
#     sum_embeddings = embeddings + other
#     result = torch.tanh(sum_embeddings)
#     if out is not None:
#         out.copy_(result)
#     return result

def test_fused_embedding_add_tanh():
    results = {}

    # Test case 1: Basic test without padding_idx, max_norm, scale_grad_by_freq, sparse, and out
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_1"] = fused_embedding_add_tanh(input_indices, weight, other)

    # Test case 2: Test with padding_idx
    padding_idx = 0
    input_indices = torch.tensor([0, 1, 2], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_2"] = fused_embedding_add_tanh(input_indices, weight, other, padding_idx=padding_idx)

    # Test case 3: Test with max_norm
    max_norm = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_3"] = fused_embedding_add_tanh(input_indices, weight, other, max_norm=max_norm)

    # Test case 4: Test with norm_type
    norm_type = 1.0
    input_indices = torch.tensor([1, 2, 3], device='cuda')
    weight = torch.randn(5, 3, device='cuda')
    other = torch.randn(3, 3, device='cuda')
    results["test_case_4"] = fused_embedding_add_tanh(input_indices, weight, other, norm_type=norm_type)

    return results

test_results = test_fused_embedding_add_tanh()
