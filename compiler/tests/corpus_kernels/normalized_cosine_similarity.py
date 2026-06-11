import triton
import triton.language as tl


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


