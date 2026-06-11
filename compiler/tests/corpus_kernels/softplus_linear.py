import triton
import triton.language as tl


@triton.jit
def _softplus_linear_kernel(
    out_ptr,
    input_ptr,
    linear_ptr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(linear_ptr + offsets, mask=mask, other=0.0)
    
    # Softplus: log(1 + exp(beta * x)) for x < threshold, else beta * x
    scaled = beta * x
    condition = scaled < threshold
    result = tl.where(
        condition,
        (1.0 / beta) * tl.log(1.0 + tl.exp(scaled)),
        x
    )
    
    tl.store(out_ptr + offsets, result, mask=mask)

