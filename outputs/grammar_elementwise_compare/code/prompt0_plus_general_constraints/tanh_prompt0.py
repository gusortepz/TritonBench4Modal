import triton
import triton.language as tl
import torch

@triton.jit
def _tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Create block indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for elements that are out of bounds
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute tanh
    output_data = tl.math.tanh(input_data)
    
    # Store output data
    tl.store(output_ptr + offsets, output_data, mask=mask)

def tanh(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the hyperbolic tangent of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.
        
    Returns:
        Tensor: The output tensor containing the hyperbolic tangent of the input elements.
    """
    # Ensure input is contiguous and on the correct device
    if not input.is_contiguous():
        input = input.contiguous()
        
    device = input.device
    dtype = input.dtype
    n_elements = input.numel()
    
    # Determine output tensor
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(f"out must have the same shape as input, got {out.shape} and {input.shape}")
        if not out.is_contiguous():
            out = out.contiguous()
        output = out
    else:
        output = torch.empty_like(input)
        
    # Grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Determine block size based on input size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Launch kernel
    _tanh_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
