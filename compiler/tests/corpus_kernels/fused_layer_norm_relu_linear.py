import triton
import triton.language as tl


@triton.jit
def _fused_relu_ln_kernel(
    output_ptr,
    input_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    col_start = tl.arange(0, BLOCK_SIZE)
    
    if row_idx >= N:
        return
    
    # Load input (post-ReLU linear output)
    offsets = row_idx * M + col_start
    mask = col_start < M
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and reciprocal std for layer normalization
    mean = tl.sum(x, axis=0) / M
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / M
    rstd = tl.rsqrt(var + eps)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Apply affine if present
    if has_weight and has_bias:
        w = tl.load(weight_ptr + col_start, mask=mask, other=0.0)
        b = tl.load(bias_ptr + col_start, mask=mask, other=0.0)
        y = x_norm * w + b
    elif has_weight:
        w = tl.load(weight_ptr + col_start, mask=mask, other=0.0)
        y = x_norm * w
    elif has_bias:
        b = tl.load(bias_ptr + col_start, mask=mask, other=0.0)
        y = x_norm + b
    else:
        y = x_norm
    
    # Store output, mean, and rstd
    tl.store(output_ptr + offsets, y, mask=mask)
    if col_start == 0:
        tl.store(mean_ptr + row_idx, mean)
        tl.store(rstd_ptr + row_idx, rstd)

