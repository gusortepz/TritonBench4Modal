import torch
import triton
import triton.language as tl

@triton.jit
def relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Create block indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    relu_data = tl.maximum(input_data, 0.0)
    
    # Apply square root
    output_data = tl.sqrt(relu_data)
    
    # Store output
    tl.store(output_ptr + offsets, output_data, mask=mask)

def relu_sqrt(input: torch.Tensor, inplace: bool = False, out: torch.Tensor = None) -> torch.Tensor:
    """
    Applies the rectified linear unit (ReLU) function to each element in input,
    and then computes the square root of the result.
    
    Args:
        input (torch.Tensor): The input tensor.
        inplace (bool, optional): If True, modifies input in-place (if possible). Default is False.
        out (torch.Tensor, optional): The output tensor.
        
    Returns:
        torch.Tensor: The output tensor.
    """
    # Ensure input is contiguous
    if not input.is_contiguous():
        input = input.contiguous()
        
    n_elements = input.numel()
    
    # Determine output tensor
    if out is not None:
        if out.numel() != n_elements:
            raise ValueError("Output tensor size does not match input tensor size.")
        output = out
    else:
        output = torch.empty_like(input)
        
    # Determine if we can do in-place operation
    if inplace and input.data_ptr() == output.data_ptr():
        # In-place operation
        # We need to handle the case where input and output are the same tensor
        # For simplicity, we'll just use the same pointer logic
        pass
        
    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Choose block size based on input size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Launch kernel
    relu_sqrt_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
