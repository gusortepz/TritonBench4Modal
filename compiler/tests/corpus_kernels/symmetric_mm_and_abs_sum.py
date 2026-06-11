import triton
import triton.language as tl


@triton.jit
def _abs_sum_kernel(y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Kernel to compute sum of absolute values of a matrix."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n)
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    abs_y = tl.abs(y)
    result = tl.sum(abs_y)
    
    tl.atomic_add(y_ptr + n, result)


