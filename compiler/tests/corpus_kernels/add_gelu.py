import triton
import triton.language as tl


@triton.jit
def _add_gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    numel,
    alpha: tl.constexpr,
    is_other_scalar: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if is_other_scalar:
        other_val = tl.load(other_ptr)
        y = x + alpha * other_val
    else:
        other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        y = x + alpha * other_val

    if approximate == "tanh":
        cdf = 0.5 * (
            1.0
            + (
                2.0
                * tl.sigmoid(
                    2.0 * (y + 0.044715 * y * y * y) * 0.7978845608028654
                )
                - 1.0
            )
        )
        result = y * cdf
    else:
        result = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


