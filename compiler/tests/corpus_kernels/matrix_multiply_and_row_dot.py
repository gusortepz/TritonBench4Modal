import triton
import triton.language as tl


@triton.jit
def _row_dot_kernel(
    row1_ptr,
    row2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute dot product of two rows using Triton."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    row1 = tl.load(row1_ptr + offsets, mask=mask, other=0.0)
    row2 = tl.load(row2_ptr + offsets, mask=mask, other=0.0)

    product = row1 * row2
    sum_val = tl.sum(product)

    tl.atomic_add(out_ptr, sum_val)


