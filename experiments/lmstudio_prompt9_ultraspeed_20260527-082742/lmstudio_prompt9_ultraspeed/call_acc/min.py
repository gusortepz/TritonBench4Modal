import torch
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@triton.jit
def _min_kernel(x_ptr, y_ptr, idx_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * n
    min_val = tl.full((1,), float('inf'), dtype=x.dtype)
    min_idx = tl.full((1,), -1, dtype=tl.int32)
    
    for start in range(0, n, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=float('inf'))
        
        local_min = tl.min(x)
        local_idx = tl.argmin(x)
        
        min_val = tl.where(local_min < min_val, local_min, min_val)
        min_idx = tl.where(local_min < min_val, local_idx, min_idx)
        
    tl.store(y_ptr + pid, min_val)
    tl.store(idx_ptr + pid, min_idx)

def min(input, dim, keepdim=False, *, out=None):
    if not input.is_cuda or not input.is_floating_point():
        return torch.min(input, dim=dim, keepdim=keepdim, out=out)
        
    if dim < 0:
        dim = input.dim() + dim
        
    x = input.contiguous()
    n = x.shape[dim]
    other_dims = x.shape[:dim] + x.shape[dim+1:]
    m = x.numel() // n if n > 0 else 1
    
    if keepdim:
        out_shape = list(other_dims) + [1]
    else:
        out_shape = list(other_dims)
        
    if out is not None:
        out_min, out_idx = out
        if out_min.shape != tuple(out_shape) or out_idx.shape != tuple(out_shape):
            return torch.min(input, dim=dim, keepdim=keepdim, out=out)
        out_min = out_min.contiguous()
        out_idx = out_idx.contiguous()
    else:
        out_min = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        out_idx = torch.empty(out_shape, dtype=torch.long, device=x.device)
        
    if n == 0 or m == 0:
        out_min.zero_()
        out_idx.fill_(-1)
        return out_min, out_idx
        
    BLOCK_SIZE = 1024
    if n > BLOCK_SIZE:
        BLOCK_SIZE = 1 << (n-1).bit_length()
        
    grid = (m,)
    _min_kernel[grid](x, out_min, out_idx, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return out_min, out_idx

##################################################################################################################################################



import torch

def test_min():
    results = {}

    # Test case 1: 2D tensor, dim=0, keepdim=False
    input_tensor = torch.tensor([[1, 2, 3], [4, 0, 6]], device='cuda')
    results["test_case_1"] = min(input_tensor, dim=0)

    # Test case 2: 2D tensor, dim=1, keepdim=False
    input_tensor = torch.tensor([[1, 2, 3], [4, 0, 6]], device='cuda')
    results["test_case_2"] = min(input_tensor, dim=1)

    # Test case 3: 3D tensor, dim=2, keepdim=True
    input_tensor = torch.tensor([[[1, 2, 3], [4, 0, 6]], [[7, 8, 9], [10, 11, 12]]], device='cuda')
    results["test_case_3"] = min(input_tensor, dim=2, keepdim=True)

    # Test case 4: 1D tensor, dim=0, keepdim=False
    input_tensor = torch.tensor([1, 2, 3, 0, 4, 5], device='cuda')
    results["test_case_4"] = min(input_tensor, dim=0)

    return results

test_results = test_min()
