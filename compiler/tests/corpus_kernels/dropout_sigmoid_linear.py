import triton
import triton.language as tl


@triton.jit
def _sigmoid_dropout_kernel(output_ptr, output_stride, numel, p_keep: tl.constexpr, philox_seed, philox_offset):
    """Fused sigmoid + dropout kernel."""
    idx = triton.program_id(0) * triton.next_power_of_2(1024) + tl.arange(0, triton.next_power_of_2(1024))
    mask = idx < numel
    
    offset = philox_offset + idx
    rand_vals = tl.rand(philox_seed, offset)
    keep_mask = rand_vals < p_keep
    
    x = tl.load(output_ptr + idx * output_stride, mask=mask, other=0.0)
    sig_x = 1.0 / (1.0 + tl.exp(-x))
    result = sig_x * keep_mask.to(sig_x.dtype) / p_keep
    
    tl.store(output_ptr + idx * output_stride, result, mask=mask)


