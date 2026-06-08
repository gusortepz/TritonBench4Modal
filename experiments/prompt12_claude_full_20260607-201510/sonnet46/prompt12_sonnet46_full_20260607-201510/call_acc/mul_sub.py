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
def _mul_sub_kernel(
    input_ptr,
    other_mul_ptr,
    other_sub_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_mul_ptr + offsets, mask=mask)
    z = tl.load(other_sub_ptr + offsets, mask=mask)

    result = x * y - alpha * z

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_sub_scalar_mul_kernel(
    input_ptr,
    other_sub_ptr,
    output_ptr,
    n_elements,
    scalar_mul,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    z = tl.load(other_sub_ptr + offsets, mask=mask)

    result = x * scalar_mul - alpha * z

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_sub_scalar_sub_kernel(
    input_ptr,
    other_mul_ptr,
    output_ptr,
    n_elements,
    scalar_sub,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_mul_ptr + offsets, mask=mask)

    result = x * y - alpha * scalar_sub

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_sub_both_scalar_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar_mul,
    scalar_sub,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    result = x * scalar_mul - alpha * scalar_sub

    tl.store(output_ptr + offsets, result, mask=mask)


def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> Tensor:
    # Check if input is CUDA tensor and eligible for Triton
    if (
        isinstance(input, Tensor)
        and input.is_cuda
        and input.is_contiguous()
        and input.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    ):
        # Check if other_mul and other_sub are compatible for Triton
        mul_is_tensor = isinstance(other_mul, Tensor)
        sub_is_tensor = isinstance(other_sub, Tensor)

        mul_compatible = (not mul_is_tensor) or (
            mul_is_tensor
            and other_mul.is_cuda
            and other_mul.is_contiguous()
            and other_mul.shape == input.shape
            and other_mul.dtype == input.dtype
        )
        sub_compatible = (not sub_is_tensor) or (
            sub_is_tensor
            and other_sub.is_cuda
            and other_sub.is_contiguous()
            and other_sub.shape == input.shape
            and other_sub.dtype == input.dtype
        )

        if mul_compatible and sub_compatible:
            try:
                n_elements = input.numel()
                BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                result = torch.empty_like(input)
                alpha_val = float(alpha)

                if mul_is_tensor and sub_is_tensor:
                    _mul_sub_kernel[grid](
                        input,
                        other_mul,
                        other_sub,
                        result,
                        n_elements,
                        alpha_val,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                elif not mul_is_tensor and sub_is_tensor:
                    _mul_sub_scalar_mul_kernel[grid](
                        input,
                        other_sub,
                        result,
                        n_elements,
                        float(other_mul),
                        alpha_val,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                elif mul_is_tensor and not sub_is_tensor:
                    _mul_sub_scalar_sub_kernel[grid](
                        input,
                        other_mul,
                        result,
                        n_elements,
                        float(other_sub),
                        alpha_val,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                else:
                    _mul_sub_both_scalar_kernel[grid](
                        input,
                        result,
                        n_elements,
                        float(other_mul),
                        float(other_sub),
                        alpha_val,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )

                if out is not None:
                    out.copy_(result)
                    return out
                return result
            except Exception:
                pass

    # PyTorch fallback
    result = input * other_mul - alpha * other_sub
    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch

def test_mul_sub():
    results = {}

    # Test case 1: input, other_mul, other_sub are tensors
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_mul_tensor = torch.tensor([0.5, 1.5, 2.5], device='cuda')
    other_sub_tensor = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    results["test_case_1"] = mul_sub(input_tensor, other_mul_tensor, other_sub_tensor)

    # Test case 2: input is a tensor, other_mul is a number, other_sub is a tensor
    other_mul_number = 2.0
    results["test_case_2"] = mul_sub(input_tensor, other_mul_number, other_sub_tensor)

    # Test case 3: input is a tensor, other_mul is a tensor, other_sub is a number
    other_sub_number = 0.5
    results["test_case_3"] = mul_sub(input_tensor, other_mul_tensor, other_sub_number)

    # Test case 4: input, other_mul, other_sub are numbers
    input_number = 3.0
    results["test_case_4"] = mul_sub(input_number, other_mul_number, other_sub_number)

    return results

test_results = test_mul_sub()
