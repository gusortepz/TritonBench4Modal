import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    approximate: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if approximate == 0:
        cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
        y = x * cdf
    else:
        tanh_arg = 2.0 * (x + 0.044715 * x * x * x) * 0.7978845608
        tanh_result = 2.0 * tl.sigmoid(tanh_arg) - 1.0
        y = 0.5 * x * (1.0 + tanh_result)

    tl.store(output_ptr + offsets, y, mask=mask)


