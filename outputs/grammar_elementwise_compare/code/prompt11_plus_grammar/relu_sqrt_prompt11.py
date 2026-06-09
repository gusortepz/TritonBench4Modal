import torch
import triton
import triton.language as tl

@triton.jit
def _relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    relu = tl.maximum(x, 0.0)
    result = tl.sqrt(relu)
    tl.store(output_ptr + offsets, result, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    if input.dtype != torch.float32 and input.dtype != torch.float64:
        result = torch.sqrt(torch.relu(input.float()))
        if out is not None:
            out.copy_(result)
            return out
        return result
    if inplace:
        output = input
    elif out is None:
        output = torch.empty_like(input)
    else:
        output = out
    n_elements = input.numel()
    if n_elements == 0:
        return output
    input_contig = input.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _relu_sqrt_kernel[grid](input_contig, output, n_elements, BLOCK_SIZE=1024)
    return output
