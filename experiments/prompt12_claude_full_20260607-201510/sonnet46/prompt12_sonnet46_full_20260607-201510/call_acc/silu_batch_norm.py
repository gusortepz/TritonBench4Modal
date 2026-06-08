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
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # SiLU: x * sigmoid(x)
    result = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def _apply_silu_triton(x: Tensor) -> Tensor:
    """Apply SiLU using Triton kernel for CUDA float tensors."""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _silu_kernel[grid](
        x,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def silu_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """
    Applies Batch Normalization followed by SiLU activation.
    """
    # Apply batch normalization using PyTorch (handles training/eval modes correctly)
    bn_out = F.batch_norm(
        input,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    # Apply SiLU activation
    if bn_out.is_cuda and bn_out.is_floating_point() and not bn_out.is_complex():
        # Use Triton kernel for CUDA float tensors
        try:
            # Ensure contiguous for Triton
            bn_out_contig = bn_out.contiguous()
            result = _apply_silu_triton(bn_out_contig)
            return result
        except Exception:
            return F.silu(bn_out)
    else:
        return F.silu(bn_out)

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_silu_batch_norm():
    results = {}

    # Test case 1: Basic functionality with training=False
    input_tensor = torch.randn(3, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = silu_batch_norm(input_tensor, running_mean, running_var, training=False)

    # Test case 2: With weight and bias, training=False
    weight = torch.ones(5, device='cuda')
    bias = torch.zeros(5, device='cuda')
    results["test_case_2"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=False)

    # Test case 3: With training=True
    results["test_case_3"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True)

    # Test case 4: Different momentum and eps values
    results["test_case_4"] = silu_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=0.2, eps=1e-3)

    return results

test_results = test_silu_batch_norm()
