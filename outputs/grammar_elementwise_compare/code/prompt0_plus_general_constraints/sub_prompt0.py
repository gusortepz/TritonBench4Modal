import triton
import triton.language as tl
import torch

@triton.jit
def _sub_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    # Create block indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for the current block
    mask = offsets < n_elements
    
    # Load input and other tensors
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Perform subtraction: out = input - alpha * other
    output_val = input_val - alpha * other_val
    
    # Store result
    tl.store(output_ptr + offsets, output_val, mask=mask)

def sub(input: torch.Tensor, other: torch.Tensor, *, alpha: float = 1.0, out: torch.Tensor = None) -> torch.Tensor:
    """
    Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.
    The operation is defined as: out_i = input_i - alpha * other_i.
    
    Args:
        input (Tensor): the input tensor.
        other (Tensor or Number): the tensor or number to subtract from input.
        alpha (Number): the multiplier for other.
        out (Tensor, optional): the output tensor.
        
    Returns:
        Tensor: The output tensor.
    """
    # Handle broadcasting and type promotion
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast other to input's shape
    if other.shape != input.shape:
        other = other.broadcast_to(input.shape)
    
    # Ensure input and other are contiguous
    input = input.contiguous()
    other = other.contiguous()
    
    # Determine output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.contiguous()
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape.")
    
    # Grid size
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _sub_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        alpha=alpha,
        BLOCK_SIZE=1024
    )
    
    return out
