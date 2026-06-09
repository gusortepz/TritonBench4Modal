import torch
import triton
import triton.language as tl

@triton.jit
def _tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0
    tl.store(output_ptr + offsets, result, mask=mask)

def tanh(input, *, out=None):
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
    n_elements = input.numel()
    if n_elements == 0:
        return output
    input_contig = input.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _tanh_kernel[grid](input_contig, output, n_elements, BLOCK_SIZE=128)
    return output
