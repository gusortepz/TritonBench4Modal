import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 0:
        # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt2_inv = 0.7071067811865476
        y = 0.5 * x * (1.0 + tl.erf(x * sqrt2_inv))
    else:
        # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * (x + 0.044715 * x * x * x) * 0.7978845608) - 1.0))
        y = cdf
    
    tl.store(output_ptr + offsets, y, mask=mask)

