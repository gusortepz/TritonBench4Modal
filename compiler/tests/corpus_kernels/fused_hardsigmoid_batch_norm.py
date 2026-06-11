import triton
import triton.language as tl


@triton.jit
def _hardsigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # hardsigmoid(x) = clip(x + 3, 0, 6) / 6
    y = (x + 3.0) / 6.0
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    
    tl.store(y_ptr + offsets, y, mask=mask)

