import triton
import triton.language as tl


@triton.jit
def _signbit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute signbit: check if the sign bit is set
    # For IEEE 754 floats, signbit is 1 if x < 0 or (x == 0 and sign(x) is negative)
    # We use tl.where to handle both cases correctly
    signbit_result = tl.where(x < 0.0, 1.0, 0.0)
    # Handle the case of negative zero: if x == 0, check if it's negative zero
    is_zero = x == 0.0
    # For negative zero, we need to check the sign bit directly
    # In IEEE 754, we can use a bitwise operation, but tl doesn't expose that directly
    # Instead, we handle it by checking if (1.0 / x) is negative infinity
    neg_inf = tl.where(is_zero, tl.where(1.0 / x < 0.0, 1.0, 0.0), signbit_result)

    # Store the result
    tl.store(output_ptr + offsets, neg_inf, mask=mask)

