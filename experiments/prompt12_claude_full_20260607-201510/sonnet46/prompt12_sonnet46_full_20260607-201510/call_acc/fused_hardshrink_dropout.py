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
def _fused_dropout_hardshrink_kernel(
    input_ptr,
    output_ptr,
    seed,
    n_elements,
    p: tl.constexpr,
    lambd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Dropout using Triton's random number generator
    # Generate random numbers for dropout
    rand = tl.rand(seed, offsets)
    # Keep probability is (1 - p), so zero out where rand < p
    keep = rand >= p
    # Scale by 1/(1-p) for training (inverted dropout)
    scale = 1.0 / (1.0 - p)
    x = tl.where(keep, x * scale, 0.0)

    # Hard shrinkage: output = x if |x| > lambd else 0
    abs_x = tl.abs(x)
    x = tl.where(abs_x > lambd, x, 0.0)

    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def _hardshrink_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    lambd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Hard shrinkage: output = x if |x| > lambd else 0
    abs_x = tl.abs(x)
    x = tl.where(abs_x > lambd, x, 0.0)

    tl.store(output_ptr + offsets, x, mask=mask)


def fused_hardshrink_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> torch.Tensor:
    # Validate p
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")

    # If not CUDA or complex dtype, fall back to PyTorch
    if not input.is_cuda or not input.is_floating_point():
        # Apply dropout
        dropped = F.dropout(input, p=p, training=training, inplace=inplace)
        # Apply hard shrinkage
        return F.hardshrink(dropped, lambd=lambd)

    # Contiguous input for Triton
    x = input.contiguous()
    n_elements = x.numel()

    if n_elements == 0:
        return input.clone() if not inplace else input

    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if training and p > 0.0:
        # Check for edge case where p == 1.0
        if p == 1.0:
            return torch.zeros_like(input)

        # Use fused Triton kernel for dropout + hardshrink
        output = torch.empty_like(x)
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32).item()

        try:
            _fused_dropout_hardshrink_kernel[grid](
                x,
                output,
                seed,
                n_elements,
                p,
                lambd,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return output
        except Exception:
            # Fallback to PyTorch
            dropped = F.dropout(input, p=p, training=training, inplace=inplace)
            return F.hardshrink(dropped, lambd=lambd)
    else:
        # No dropout (training=False or p=0), just apply hardshrink
        output = torch.empty_like(x)
        try:
            _hardshrink_kernel[grid](
                x,
                output,
                n_elements,
                lambd,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return output
        except Exception:
            return F.hardshrink(x, lambd=lambd)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_hardshrink_dropout(input: torch.Tensor, p: float=0.5, training: bool=True, inplace: bool=False, lambd: float=0.5) -> torch.Tensor:
#     """
#     Applies a fused operation consisting of dropout followed by hard shrinkage on the input tensor.

#     Args:
#         input (Tensor): The input tensor.
#         p (float, optional): Probability of an element to be zeroed in dropout. Default is 0.5.
#         training (bool, optional): Apply dropout if True. Default is True.
#         inplace (bool, optional): If set to True, dropout will be applied in-place. Default is False.
#         lambd (float, optional): The lambda parameter for the hard shrinkage function. Default is 0.5.

#     Returns:
#         Tensor: Result after applying dropout and then hard shrinkage on the input.
#     """
#     if training:
#         input = F.dropout(input, p=p, training=training, inplace=inplace)
#     return F.hardshrink(input, lambd)

def test_fused_hardshrink_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_1"] = fused_hardshrink_dropout(input_tensor)
    
    # Test case 2: Dropout with p=0.3
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_2"] = fused_hardshrink_dropout(input_tensor, p=0.3)
    
    # Test case 3: Dropout with training=False
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_3"] = fused_hardshrink_dropout(input_tensor, training=False)
    
    # Test case 4: Hard shrinkage with lambd=0.7
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_4"] = fused_hardshrink_dropout(input_tensor, lambd=0.7)
    
    return results

test_results = test_fused_hardshrink_dropout()
