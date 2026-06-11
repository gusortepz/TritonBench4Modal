import triton
import triton.language as tl


@triton.jit
def _log_tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    log_x = tl.log(x)
    tanh_log_x = 2.0 * tl.sigmoid(2.0 * log_x) - 1.0
    
    tl.store(output_ptr + offsets, tanh_log_x, mask=mask)


