import triton
import triton.language as tl


@triton.jit
def _rad2deg_sqrt_kernel(
    input_ptr,
    deg_ptr,
    sqrt_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Convert radians to degrees: x * (180 / pi)
    deg_result = x * (180.0 / 3.141592653589793)

    # Compute square root
    sqrt_result = tl.sqrt(x)

    tl.store(deg_ptr + offsets, deg_result, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_result, mask=mask)

