import triton
import triton.language as tl


@triton.jit
def _fused_hardshrink_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    p: tl.constexpr,
    training: tl.constexpr,
    lambd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if training:
        dropout_mask = tl.rand(tl.uint32(offsets), n_elements) > p
        scale = 1.0 / (1.0 - p)
        x = tl.where(dropout_mask, x * scale, 0.0)
    
    abs_x = tl.abs(x)
    hard_shrink = tl.where(abs_x > lambd, x, 0.0)
    
    tl.store(output_ptr + offsets, hard_shrink, mask=mask)

