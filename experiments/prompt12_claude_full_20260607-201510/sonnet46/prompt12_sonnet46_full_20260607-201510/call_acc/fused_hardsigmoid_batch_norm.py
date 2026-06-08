import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
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
def _hardsigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # hardsigmoid: relu6(x + 3) / 6
    # = clamp(x/6 + 0.5, 0, 1)
    y = x / 6.0 + 0.5
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def _apply_hardsigmoid_triton(x: Tensor) -> Tensor:
    if not x.is_cuda or not x.is_contiguous():
        return F.hardsigmoid(x)
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _hardsigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False,
) -> torch.Tensor:
    # Apply batch normalization
    bn_out = F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    # Apply hardsigmoid
    if bn_out.is_cuda and bn_out.dtype in (torch.float32, torch.float16, torch.bfloat16):
        if not bn_out.is_contiguous():
            bn_out = bn_out.contiguous()
        try:
            result = _apply_hardsigmoid_triton(bn_out)
            if inplace:
                x.copy_(result) if x.shape == result.shape else None
            return result
        except Exception:
            pass

    # Fallback
    return F.hardsigmoid(bn_out, inplace=inplace)

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
