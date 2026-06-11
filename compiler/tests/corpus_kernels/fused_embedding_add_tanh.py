import triton
import triton.language as tl


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


