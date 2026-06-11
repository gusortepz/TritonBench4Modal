import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(
    input_ptr,
    output_ptr,
    numel,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 1:
        # GELU tanh approximation
        cdf = 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        y = x * cdf
    else:
        # GELU exact
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    
    tl.store(output_ptr + offsets, y, mask=mask)


