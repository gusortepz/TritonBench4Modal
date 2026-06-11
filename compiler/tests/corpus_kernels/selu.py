import triton
import triton.language as tl


@triton.jit
def _selu_kernel(
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
    
    # SELU constants
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    
    # SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(0.0, alpha * (tl.exp(x) - 1.0))
    y = scale * (pos + neg)
    
    tl.store(output_ptr + offsets, y, mask=mask)


