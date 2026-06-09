import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    input_ptr, 
    other_ptr, 
    output_ptr, 
    n_elements, 
    alpha, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load other (scalar or tensor)
    # If other is a scalar, other_ptr might be a pointer to a single value or we handle it differently
    # For simplicity in this generic wrapper, we assume 'other' is broadcasted or a scalar.
    # If 'other' is a tensor, we need to handle broadcasting logic or assume it's already aligned.
    # Given the complexity of general broadcasting in a single kernel without metadata, 
    # we will assume 'other' is either a scalar or a tensor of the same shape for this specific benchmark task 
    # or handle the scalar case explicitly.
    
    # Let's assume 'other' is passed as a pointer. If it's a scalar, we load it once.
    # However, Triton kernels usually operate on tensors. 
    # To support 'other' being a number, we can pass it as a constant or a pointer to a scalar.
    # Let's assume 'other' is a tensor for the kernel, and the wrapper handles scalar conversion.
    
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Compute result
    result = input_val + alpha * other_val
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Handle device and dtype
    device = input.device
    dtype = input.dtype
    
    # Handle 'other' being a scalar
    if isinstance(other, (int, float, complex)):
        # Convert to tensor if needed for broadcasting or just use scalar
        # For Triton, we can pass the scalar value directly if it's a constant, 
        # but here we treat it as a broadcasted tensor for generality in the wrapper logic.
        # However, to keep the kernel simple, we can just multiply the scalar alpha by the scalar other.
        # But the kernel expects pointers. 
        # Let's create a dummy tensor for 'other' if it's a scalar to reuse the kernel, 
        # or handle it in the wrapper.
        
        # Actually, a better approach for a generic wrapper is to ensure 'other' is a tensor.
        if not isinstance(other, torch.Tensor):
            other = torch.tensor(other, device=device, dtype=dtype)
    
    # Ensure 'other' is on the same device and has compatible dtype
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=device, dtype=dtype)
    
    # Broadcast 'other' to 'input' shape if necessary
    # Triton doesn't automatically broadcast in the kernel for arbitrary shapes without metadata.
    # We will rely on PyTorch's broadcasting for the kernel launch if shapes are compatible,
    # or we can just assume the user passes compatible shapes for this benchmark.
    # However, to be robust, we can broadcast 'other' to 'input' shape here.
    other = torch.broadcast_to(other, input.shape)
    
    # Determine output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output shape must match input shape"
        assert out.device == input.device, "Output device must match input device"
    
    # Grid size
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](
        input, 
        other, 
        out, 
        n_elements, 
        alpha, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
