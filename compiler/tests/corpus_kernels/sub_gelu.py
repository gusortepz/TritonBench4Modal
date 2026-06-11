import triton
import triton.language as tl


@triton.jit
def _sub_gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Sub: x - alpha * other
    result = x - alpha * other_val
    
    # GELU
    if approximate == "tanh":
        # GELU (tanh approximation):
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        SQRT_2_OVER_PI = 0.7978845608028654
        COEFF = 0.044715
        x_cubed = result * result * result
        arg = SQRT_2_OVER_PI * (result + COEFF * x_cubed)
        tanh_val = 2.0 * tl.sigmoid(2.0 * arg) - 1.0
        gelu_result = 0.5 * result * (1.0 + tanh_val)
    else:
        # GELU (exact):
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_result = 0.5 * result * (1.0 + tl.erf(result * 0.7071067811865476))
    
    tl.store(output_ptr + offsets, gelu_result, mask=mask)


