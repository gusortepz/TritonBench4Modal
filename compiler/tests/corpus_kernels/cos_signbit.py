import triton
import triton.language as tl


@triton.jit
def _cos_signbit_kernel(
    input_ptr,
    cos_ptr,
    signbit_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    cos_val = tl.cos(x)
    signbit_val = tl.where(cos_val < 0.0, 1, 0)
    
    tl.store(cos_ptr + offsets, cos_val, mask=mask)
    tl.store(signbit_ptr + offsets, signbit_val, mask=mask)


