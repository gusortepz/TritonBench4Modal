import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise add kernel: output = input + alpha * other"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    
    z = x + alpha * y
    
    tl.store(output_ptr + offsets, z, mask=mask)


