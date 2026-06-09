import triton
import triton.language as tl
import torch

@triton.jit
def _sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Create block IDs
    block_id = tl.program_id(0)
    # Compute offsets for this block
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for this block
    mask = offsets < n_elements
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute square root
    output_data = tl.sqrt(input_data)
    # Store output data
    tl.store(output_ptr + offsets, output_data, mask=mask)

def sqrt(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the square-root of the elements of the input tensor.
    It computes the square root element-wise.
    
    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.
        
    Returns:
        Tensor: the output tensor with square root of input elements.
    """
    # Ensure input is contiguous
    if not input.is_contiguous():
        input = input.contiguous()
        
    # Determine output tensor
    if out is not None:
        output = out
        # Ensure output is contiguous
        if not output.is_contiguous():
            output = output.contiguous()
    else:
        output = torch.empty_like(input)
        
    # Check if input is empty
    if input.numel() == 0:
        return output
        
    # Grid configuration
    n_elements = input.numel()
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Choose block size based on input size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Launch kernel
    _sqrt_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
