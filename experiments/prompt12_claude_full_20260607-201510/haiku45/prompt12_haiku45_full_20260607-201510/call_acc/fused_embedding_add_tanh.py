import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
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
def _fused_embedding_add_tanh_kernel(
    embeddings_ptr,
    other_ptr,
    output_ptr,
    numel: tl.constexpr,
    block_size: tl.constexpr,
):
    """Fused kernel for embedding + other + tanh."""
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel

    embed_vals = tl.load(embeddings_ptr + offsets, mask=mask, other=0.0)
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    added = embed_vals + other_vals
    result = 2.0 * tl.sigmoid(2.0 * added) - 1.0

    tl.store(output_ptr + offsets, result, mask=mask)


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
    """
    Fused operation combining embedding lookup, element-wise addition, and tanh activation.

    Args:
        input_indices: LongTensor of arbitrary shape containing indices into the embedding matrix.
        weight: Embedding matrix of shape (V, D).
        other: Tensor to be added to embeddings, broadcastable to embedding shape.
        padding_idx: If specified, entries at padding_idx do not contribute to gradient.
        max_norm: If given, embedding vectors with norm > max_norm are renormalized.
        norm_type: The p-norm to compute for max_norm option. Default: 2.0.
        scale_grad_by_freq: If True, scale gradients by inverse frequency. Default: False.
        sparse: If True, gradient w.r.t. weight will be sparse. Default: False.
        out: Output tensor. Ignored if None.

    Returns:
        Output tensor of shape (*) + (D,) where (*) is the shape of input_indices.
    """
    embeddings = F.embedding(
        input_indices,
        weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )

    added = embeddings + other
    result = torch.tanh(added)

    if out is not None:
        out.copy_(result)
        return out
    return result

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
