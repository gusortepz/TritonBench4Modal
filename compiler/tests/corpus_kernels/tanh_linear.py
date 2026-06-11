import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(
    y_ptr,
    y_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets * y_stride, mask=mask, other=0.0)
    y_tanh = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    tl.store(y_ptr + offsets * y_stride, y_tanh, mask=mask)

