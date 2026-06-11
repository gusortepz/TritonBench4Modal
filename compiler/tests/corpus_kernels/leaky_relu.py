import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(
    output_ptr,
    input_ptr,
    numel,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0) + negative_slope * tl.minimum(x, 0.0)
    
    tl.store(output_ptr + offsets, y, mask=mask)


