import triton
import triton.language as tl


@triton.jit
def _relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    x = tl.maximum(x, 0.0)
    y = tl.sqrt(x)
    tl.store(output_ptr + offsets, y, mask=mask)


