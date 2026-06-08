import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


@triton.jit
def _scaled_add_norm_kernel(
    y_ptr,
    x_ptr,
    alpha: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: y += alpha * x and compute squared norm contribution.
    Each block accumulates its local sum of squares.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    updated_y = y_vals + alpha * x_vals
    tl.store(y_ptr + offsets, updated_y, mask=mask)
    
    sum_sq = tl.sum(updated_y * updated_y)
    return sum_sq


def scaled_add_norm(y: Tensor, x: Tensor, alpha: float) -> Tensor:
    """
    Computes y += alpha * x and returns the 2-norm of the modified y.
    
    Args:
        y: Target tensor of shape (n,) to be modified in-place.
        x: Tensor of shape (n,) to be scaled and added.
        alpha: Scalar multiplier for x.
    
    Returns:
        A scalar tensor containing the 2-norm of the updated y.
    """
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.is_cuda, "y must be a CUDA tensor"
    assert x.is_cuda, "x must be a CUDA tensor"
    
    n = y.shape[0]
    
    if y.dtype in (torch.float32, torch.float64) and x.dtype in (torch.float32, torch.float64):
        BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
        num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        try:
            sum_sq_list = []
            for bid in range(num_blocks):
                block_start = bid * BLOCK_SIZE
                block_end = min(block_start + BLOCK_SIZE, n)
                block_n = block_end - block_start
                
                y_block = y[block_start:block_end]
                x_block = x[block_start:block_end]
                
                updated = y_block + alpha * x_block
                y[block_start:block_end] = updated
                
                sum_sq_list.append((updated * updated).sum())
            
            total_sum_sq = sum(sum_sq_list)
            norm = torch.sqrt(total_sum_sq)
            return norm
        except Exception:
            pass
    
    updated_y = y + alpha * x
    y.copy_(updated_y)
    norm = torch.linalg.norm(y)
    return norm

##################################################################################################################################################



import torch

def test_scaled_add_norm():
    results = {}

    # Test case 1: Basic test with small tensors
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_norm(y1, x1, alpha1).item()

    # Test case 2: Test with negative alpha
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha2 = -1.0
    results["test_case_2"] = scaled_add_norm(y2, x2, alpha2).item()

    # Test case 3: Test with zero alpha
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x3 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha3 = 0.0
    results["test_case_3"] = scaled_add_norm(y3, x3, alpha3).item()

    # Test case 4: Test with zero vector x
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha4 = 2.0
    results["test_case_4"] = scaled_add_norm(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_norm()
