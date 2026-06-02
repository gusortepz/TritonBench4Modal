import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@triton.jit
def _max_argmax_kernel(input_ptr, output_ptr, index_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_off = pid
    if row_off < n_rows:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        vals = tl.load(input_ptr + row_off * n_cols + offs, mask=mask, other=-float('inf'))
        max_val = tl.max(vals, axis=0)
        max_idx = tl.argmax(vals, axis=0)
        tl.store(output_ptr + row_off, max_val)
        tl.store(index_ptr + row_off, max_idx)

def max(input, dim, keepdim=False, *, out=None):
    if input.dim() == 2 and dim == -1 and input.size(1) <= 1024 and input.is_cuda and input.is_floating_point():
        n_rows = input.size(0)
        n_cols = input.size(1)
        output = torch.empty(n_rows, dtype=input.dtype, device=input.device)
        indices = torch.empty(n_rows, dtype=torch.long, device=input.device)
        grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
        _max_argmax_kernel[grid](input, output, indices, n_rows, n_cols, BLOCK_SIZE=1024)
        if keepdim:
            output = output.unsqueeze(dim)
            indices = indices.unsqueeze(dim)
        if out is not None:
            out[0].copy_(output)
            out[1].copy_(indices)
            return out
        return (output, indices)
    else:
        output, indices = torch.max(input, dim=dim, keepdim=keepdim)
        if out is not None:
            out[0].copy_(output)
            out[1].copy_(indices)
            return out
        return (output, indices)

##################################################################################################################################################



import torch

def test_max():
    results = {}

    # Test case 1: Basic test with a 2D tensor
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_1'] = max(input_tensor, dim=0)

    # Test case 2: Test with keepdim=True
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_2'] = max(input_tensor, dim=1, keepdim=True)

    # Test case 3: Test with a 3D tensor
    input_tensor = torch.tensor([[[1, 3, 2], [4, 6, 5]], [[7, 9, 8], [10, 12, 11]]], device='cuda')
    results['test_case_3'] = max(input_tensor, dim=2)

    # Test case 4: Test with a negative dimension
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_4'] = max(input_tensor, dim=-1)

    return results

test_results = test_max()
