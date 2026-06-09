import triton
import triton.language as tl
import torch

@triton.jit
def rsqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Create block IDs
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute reciprocal of square root
    output_data = 1.0 / tl.sqrt(input_data)
    
    # Store output data
    tl.store(output_ptr + offsets, output_data, mask=mask)

def rsqrt(input, *, out=None):
    """
    Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.
        
    Returns:
        Tensor: the output tensor.
    """
    if out is not None:
        output = out
    else:
        output = torch.empty_like(input)
    
    n_elements = input.numel()
    
    # Determine grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    rsqrt_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return output
