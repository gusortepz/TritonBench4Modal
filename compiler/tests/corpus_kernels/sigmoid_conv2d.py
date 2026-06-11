import triton
import triton.language as tl


@triton.jit
def _sigmoid_conv2d_kernel(
    output_ptr,
    stride_output_batch,
    stride_output_channels,
    stride_output_h,
    stride_output_w,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise sigmoid on convolution output."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    output_data = tl.load(output_ptr + offsets, mask=mask)
    sigmoid_output = tl.sigmoid(output_data)
    tl.store(output_ptr + offsets, sigmoid_output, mask=mask)


