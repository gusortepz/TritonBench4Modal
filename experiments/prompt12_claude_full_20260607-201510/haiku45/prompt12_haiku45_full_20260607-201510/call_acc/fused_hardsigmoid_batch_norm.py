import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
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
def _hardsigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Applies hardsigmoid(x) = clip(x + 3, 0, 6) / 6 element-wise.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # hardsigmoid(x) = clip(x + 3, 0, 6) / 6
    y = (x + 3.0) / 6.0
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    
    tl.store(y_ptr + offsets, y, mask=mask)


def fused_hardsigmoid_batch_norm(
    x: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False,
) -> Tensor:
    """
    Applies Batch Normalization followed by Hardsigmoid activation.
    
    Args:
        x: Input tensor for batch normalization and activation.
        running_mean: The running mean buffer (persistent).
        running_var: The running variance buffer (persistent).
        weight: Learnable weight of size C for the normalized tensor. Default: None
        bias: Learnable bias of size C for the normalized tensor. Default: None
        training: Flag for training mode, used to update running estimates. Default: False
        momentum: The value for the running mean and variance momentum. Default: 0.1
        eps: Small constant added to variance to improve numerical stability. Default: 1e-5
        inplace: If True, perform Hardsigmoid in-place. Default: False
    
    Returns:
        Output tensor after batch normalization and hardsigmoid activation.
    """
    
    # Apply batch normalization
    # For batch_norm, we need to infer normalized_shape from x
    if x.dim() < 2:
        raise ValueError(f"Expected at least 2D input, got {x.dim()}D")
    
    # normalized_shape is the feature dimension (channels)
    normalized_shape = x.shape[1]
    
    # Validate weight and bias shapes if provided
    w = None
    b = None
    if weight is not None and weight.shape == torch.Size([normalized_shape]):
        w = weight
    if bias is not None and bias.shape == torch.Size([normalized_shape]):
        b = bias
    
    # Apply batch normalization using torch.nn.functional.batch_norm
    # batch_norm expects (input, running_mean, running_var, weight, bias, training, momentum, eps)
    bn_output = F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=w,
        bias=b,
        training=training,
        momentum=momentum,
        eps=eps,
    )
    
    # Apply hardsigmoid activation
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float64, torch.float16):
        # Fall back to pure PyTorch for non-CUDA or non-float tensors
        return torch.nn.functional.hardsigmoid(bn_output, inplace=inplace)
    
    # Use Triton kernel for CUDA float tensors
    if inplace:
        y = bn_output
    else:
        y = bn_output.clone()
    
    n_elements = y.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    _hardsigmoid_kernel[grid](y, y, n_elements, BLOCK_SIZE=1024)
    
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_hardsigmoid_batch_norm(x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor=None, bias: torch.Tensor=None, training: bool=False, momentum: float=0.1, eps: float=1e-05, inplace: bool=False) -> torch.Tensor:
#     normalized_x = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
#     output = torch.nn.functional.hardsigmoid(normalized_x, inplace=inplace)
#     return output

def test_fused_hardsigmoid_batch_norm():
    results = {}
    
    # Test case 1: Default parameters
    x = torch.randn(10, 3, 32, 32, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_1"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var)
    
    # Test case 2: With weight and bias
    weight = torch.randn(3, device='cuda')
    bias = torch.randn(3, device='cuda')
    results["test_case_2"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight, bias)
    
    # Test case 3: Training mode
    results["test_case_3"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var, training=True)
    
    # Test case 4: Inplace operation
    results["test_case_4"] = fused_hardsigmoid_batch_norm(x, running_mean, running_var, inplace=True)
    
    return results

test_results = test_fused_hardsigmoid_batch_norm()
