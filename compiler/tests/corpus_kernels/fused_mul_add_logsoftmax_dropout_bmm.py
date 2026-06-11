import triton
import triton.language as tl


@triton.jit
def _fused_mul_add_logsoftmax_dropout_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
    dropout_p: tl.constexpr,
    seed: tl.constexpr,
):

    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Element-wise multiplication
    result = x * y

    # Dropout during training: scale by 1/(1-p) to keep expected value
    if dropout_p > 0.0:
        # Pseudo-random: approximate dropout mask
        rand_val = tl.rand(seed, offsets)
        mask_dropout = rand_val > dropout_p
        result = tl.where(mask_dropout, result / (1.0 - dropout_p), 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)

