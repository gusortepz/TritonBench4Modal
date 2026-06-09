import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(input_ptr, other_ptr, output_ptr, n_elements, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    result = x - alpha * y
    tl.store(output_ptr + offsets, result, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
    n_elements = input.numel()
    if n_elements == 0:
        return output
    if torch.is_tensor(other):
        other_tensor = other.contiguous()
    else:
        other_tensor = torch.full_like(input, other)
    input_contig = input.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _sub_kernel[grid](input_contig, other_tensor, output, n_elements, alpha, BLOCK_SIZE=1024)
    return output
