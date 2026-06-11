import triton
import triton.language as tl


@triton.jit
def _erfc_sqrt_kernel(
    input_ptr,
    erfc_ptr,
    sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    erfc_val = 1.0 - tl.erf(x)
    sqrt_val = tl.sqrt(tl.maximum(x, 0.0))

    tl.store(erfc_ptr + offsets, erfc_val, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=mask)


