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
def _logit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    has_eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if has_eps:
        x = tl.minimum(tl.maximum(x, eps), 1.0 - eps)

    # logit(x) = log(x / (1 - x)) = log(x) - log(1 - x)
    log_x = tl.log(x)
    log_1mx = tl.log(1.0 - x)
    result = log_x - log_1mx

    tl.store(output_ptr + offsets, result, mask=mask)


def logit(input: Tensor, eps: Optional[float] = None, *, out: Optional[Tensor] = None) -> Tensor:
    # Use PyTorch fallback for non-CUDA or non-float tensors
    if not input.is_cuda or not input.dtype.is_floating_point or input.is_complex():
        y = torch.logit(input, eps=eps)
        if out is not None:
            out.copy_(y)
            return out
        return y

    # Triton path for CUDA float tensors
    try:
        input_flat = input.contiguous().view(-1)
        n_elements = input_flat.numel()

        # Work in float32 for Triton kernel
        compute_dtype = input.dtype
        if compute_dtype not in (torch.float32, torch.float64, torch.float16):
            # fallback
            raise ValueError("unsupported dtype for Triton path")

        # Use float32 for computation
        x_f32 = input_flat.float()
        output_f32 = torch.empty(n_elements, dtype=torch.float32, device=input.device)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n_elements, 1)))
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        has_eps = eps is not None
        eps_val = float(eps) if has_eps else 0.0

        _logit_kernel[grid](
            x_f32,
            output_f32,
            n_elements,
            eps_val,
            has_eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Cast back to original dtype
        y = output_f32.to(compute_dtype).view(input.shape)

        if out is not None:
            out.copy_(y)
            return out
        return y
    except Exception:
        # Fallback to PyTorch
        y = torch.logit(input, eps=eps)
        if out is not None:
            out.copy_(y)
            return out
        return y

##################################################################################################################################################



import torch

def test_logit():
    results = {}

    # Test case 1: Basic test with input tensor in range [0, 1] without eps
    input1 = torch.tensor([0.2, 0.5, 0.8], device='cuda')
    results["test_case_1"] = logit(input1)

    # Test case 2: Test with input tensor in range [0, 1] with eps
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    eps = 1e-6
    results["test_case_2"] = logit(input2, eps=eps)

    # Test case 3: Test with input tensor in range [0, 1] with eps and out tensor
    input3 = torch.tensor([0.1, 0.9], device='cuda')
    out = torch.empty_like(input3)
    results["test_case_3"] = logit(input3, eps=eps, out=out)

    # Test case 4: Test with input tensor in range [0, 1] with out tensor
    input4 = torch.tensor([0.3, 0.7], device='cuda')
    out = torch.empty_like(input4)
    results["test_case_4"] = logit(input4, out=out)

    return results

test_results = test_logit()
