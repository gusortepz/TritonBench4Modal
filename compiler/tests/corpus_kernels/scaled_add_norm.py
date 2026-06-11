import triton
import triton.language as tl


@triton.jit
def _scaled_add_norm_kernel(
    y_ptr,
    x_ptr,
    alpha: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    updated_y = y_vals + alpha * x_vals
    tl.store(y_ptr + offsets, updated_y, mask=mask)
    
    sum_sq = tl.sum(updated_y * updated_y)
    return sum_sq

