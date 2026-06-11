import triton
import triton.language as tl


@triton.jit
def _log_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.log(x)
    tl.store(output_ptr + offsets, y, mask=mask)


