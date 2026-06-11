import triton
import triton.language as tl


@triton.jit
def _scaled_add_dot_kernel(
    y_ptr,
    x_ptr,
    output_ptr,
    n: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    y_updated = y + alpha * x
    tl.store(y_ptr + offsets, y_updated, mask=mask)

    dot_product = tl.sum(y_updated * y_updated)
    tl.atomic_add(output_ptr, dot_product)

