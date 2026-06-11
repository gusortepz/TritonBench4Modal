import triton
import triton.language as tl


@triton.jit
def _softmax_mul_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    dim_size: tl.constexpr,
    stride_dim: tl.constexpr,
    other_is_scalar: tl.constexpr,
    other_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if other_is_scalar:
        other_vals = other_scalar
    else:
        other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    output = input_vals * other_vals
    tl.store(output_ptr + offsets, output, mask=mask)


