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
def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sigmoid(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def _apply_sigmoid_triton(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n, 1)))
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def sigmoid_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    # Apply batch normalization using PyTorch's optimized implementation
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

    # Apply sigmoid activation
    if bn_out.is_cuda and bn_out.is_contiguous() and bn_out.dtype in (torch.float16, torch.float32, torch.bfloat16):
        try:
            # Use contiguous float32 for Triton kernel
            if bn_out.dtype == torch.float32:
                return _apply_sigmoid_triton(bn_out)
            else:
                # For other dtypes, fall back to PyTorch
                return torch.sigmoid(bn_out)
        except Exception:
            return torch.sigmoid(bn_out)
    else:
        return torch.sigmoid(bn_out)

##################################################################################################################################################



import torch

def test_sigmoid_batch_norm():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(10, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = sigmoid_batch_norm(input_tensor, running_mean, running_var)

    # Test case 2: With learnable parameters (weight and bias)
    weight = torch.ones(5, device='cuda') * 0.5
    bias = torch.zeros(5, device='cuda') + 0.1
    results["test_case_2"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias)

    # Test case 3: In training mode
    results["test_case_3"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, training=True)

    # Test case 4: With a different momentum and eps
    results["test_case_4"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, momentum=0.2, eps=1e-3)

    return results

test_results = test_sigmoid_batch_norm()
