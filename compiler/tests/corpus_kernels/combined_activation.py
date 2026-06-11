import triton
import triton.language as tl


@triton.jit
def _combined_activation_kernel(
    output_ptr,
    intermediate_ptr,
    weight2_ptr,
    bias_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load intermediate (matmul output)
    intermediate = tl.load(intermediate_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid(intermediate)
    sigmoid_x = tl.sigmoid(intermediate)

    # Compute tanh(intermediate)
    tanh_x = 2.0 * tl.sigmoid(2.0 * intermediate) - 1.0

    # Element-wise multiply: sigmoid(x) * tanh(x)
    prod = sigmoid_x * tanh_x

    # Load weight2 (broadcast if needed)
    weight2 = tl.load(weight2_ptr + offsets, mask=mask, other=1.0)

    # Load bias (broadcast if needed)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Final computation: prod * weight2 + bias
    result = prod * weight2 + bias

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

