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
def _sub_gelu_exact_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)

    z = x - alpha * y
    # GELU exact: 0.5 * z * (1 + erf(z / sqrt(2)))
    result = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_tanh_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)

    z = x - alpha * y
    # GELU tanh approximation
    # tanh(sqrt(2/pi) * (z + 0.044715 * z^3))
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * z * (1.0 + tanh_val)

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_exact_scalar_kernel(
    input_ptr,
    other_scalar,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    z = x - alpha * other_scalar
    result = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_tanh_scalar_kernel(
    input_ptr,
    other_scalar,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    z = x - alpha * other_scalar
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * z * (1.0 + tanh_val)

    tl.store(output_ptr + offsets, result, mask=mask)


def sub_gelu(input: Tensor, other, alpha=1, approximate='none', out=None) -> Tensor:
    # Check if we can use Triton fast path
    use_triton = (
        input.is_cuda
        and input.is_contiguous()
        and input.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and isinstance(approximate, str)
        and approximate in ('none', 'tanh')
    )

    if use_triton:
        is_tensor_other = isinstance(other, Tensor)
        if is_tensor_other:
            use_triton = (
                other.is_cuda
                and other.dtype == input.dtype
                and other.numel() == input.numel()
            )
        else:
            use_triton = isinstance(other, (int, float))

    if use_triton:
        n_elements = input.numel()
        output = torch.empty_like(input)
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        input_flat = input.reshape(-1)
        output_flat = output.reshape(-1)

        alpha_val = float(alpha)

        if isinstance(other, Tensor):
            other_flat = other.reshape(-1).contiguous()
            if approximate == 'tanh':
                _sub_gelu_tanh_kernel[grid](
                    input_flat, other_flat, output_flat,
                    alpha_val, n_elements, BLOCK_SIZE=BLOCK_SIZE
                )
            else:
                _sub_gelu_exact_kernel[grid](
                    input_flat, other_flat, output_flat,
                    alpha_val, n_elements, BLOCK_SIZE=BLOCK_SIZE
                )
        else:
            other_val = float(other)
            if approximate == 'tanh':
                _sub_gelu_tanh_scalar_kernel[grid](
                    input_flat, other_val, output_flat,
                    alpha_val, n_elements, BLOCK_SIZE=BLOCK_SIZE
                )
            else:
                _sub_gelu_exact_scalar_kernel[grid](
                    input_flat, other_val, output_flat,
                    alpha_val, n_elements, BLOCK_SIZE=BLOCK_SIZE
                )

        if out is not None:
            out.copy_(output)
            return out
        return output

    # PyTorch fallback
    z = torch.sub(input, other, alpha=alpha)
    result = F.gelu(z, approximate=approximate)

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_sub_gelu():
    results = {}

    # Test case 1: Basic subtraction and GELU with default approximate
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub_gelu(input_tensor, other_tensor)

    # Test case 2: Subtraction with alpha and GELU with default approximate
    alpha = 0.5
    results["test_case_2"] = sub_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Subtraction and GELU with 'tanh' approximation
    approximate = 'tanh'
    results["test_case_3"] = sub_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Subtraction with alpha and GELU with 'tanh' approximation
    results["test_case_4"] = sub_gelu(input_tensor, other_tensor, alpha=alpha, approximate=approximate)

    return results

test_results = test_sub_gelu()
