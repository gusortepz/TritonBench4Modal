import triton
import triton.language as tl


@triton.jit
def _floor_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.floor(x)
    tl.store(output_ptr + offsets, y, mask=mask)


