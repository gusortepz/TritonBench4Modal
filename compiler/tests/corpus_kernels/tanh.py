import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    tl.store(Y + offsets, y, mask=mask)


