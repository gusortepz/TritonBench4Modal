import triton
import triton.language as tl


@triton.jit
def _sqrt_tanh_kernel(
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
    
    # sqrt(x)
    sqrt_x = tl.sqrt(x)
    
    # tanh(sqrt_x) = 2*sigmoid(2*sqrt_x) - 1
    y = 2.0 * tl.sigmoid(2.0 * sqrt_x) - 1.0
    
    tl.store(output_ptr + offsets, y, mask=mask)


