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
def _add_gelu_none_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)

    z = x + alpha * y
    # GELU exact: 0.5 * z * (1 + erf(z / sqrt(2)))
    result = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _add_gelu_none_scalar_kernel(
    input_ptr,
    other_scalar,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    z = x + alpha * other_scalar
    # GELU exact
    result = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _add_gelu_tanh_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)

    z = x + alpha * y
    # GELU tanh approximation
    inner = (z + 0.044715 * z * z * z) * 0.7978845608
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * z * (1.0 + tanh_val)

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _add_gelu_tanh_scalar_kernel(
    input_ptr,
    other_scalar,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    z = x + alpha * other_scalar
    # GELU tanh approximation
    inner = (z + 0.044715 * z * z * z) * 0.7978845608
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * z * (1.0 + tanh_val)

    tl.store(output_ptr + offsets, result, mask=mask)


def _add_gelu_pytorch(input: Tensor, other, alpha=1, approximate='none') -> Tensor:
    z = torch.add(input, other, alpha=alpha)
    return F.gelu(z, approximate=approximate)


def add_gelu(input, other, alpha=1, approximate='none', out=None) -> Tensor:
    # Check if we can use Triton fast path
    use_triton = (
        isinstance(input, torch.Tensor)
        and input.is_cuda
        and input.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and (approximate in ('none', 'tanh'))
    )

    if use_triton:
        try:
            is_scalar_other = not isinstance(other, torch.Tensor)
            if not is_scalar_other:
                other_cuda = (
                    isinstance(other, torch.Tensor)
                    and other.is_cuda
                    and other.dtype == input.dtype
                    and other.shape == input.shape
                )
                if not other_cuda:
                    use_triton = False

            if use_triton:
                # Work with float32 for computation stability
                orig_dtype = input.dtype
                inp = input.contiguous()
                if inp.dtype != torch.float32:
                    inp = inp.to(torch.float32)

                output = torch.empty_like(inp)
                n_elements = inp.numel()
                BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

                if is_scalar_other:
                    scalar_val = float(other)
                    alpha_f = float(alpha)
                    if approximate == 'tanh':
                        _add_gelu_tanh_scalar_kernel[grid](
                            inp, scalar_val, output, alpha_f, n_elements, BLOCK_SIZE=BLOCK_SIZE
                        )
                    else:
                        _add_gelu_none_scalar_kernel[grid](
                            inp, scalar_val, output, alpha_f, n_elements, BLOCK_SIZE=BLOCK_SIZE
                        )
                else:
                    oth = other.contiguous()
                    if oth.dtype != torch.float32:
                        oth = oth.to(torch.float32)
                    alpha_f = float(alpha)
                    if approximate == 'tanh':
                        _add_gelu_tanh_kernel[grid](
                            inp, oth, output, alpha_f, n_elements, BLOCK_SIZE=BLOCK_SIZE
                        )
                    else:
                        _add_gelu_none_kernel[grid](
                            inp, oth, output, alpha_f, n_elements, BLOCK_SIZE=BLOCK_SIZE
                        )

                if orig_dtype != torch.float32:
                    output = output.to(orig_dtype)

                if out is not None:
                    out.copy_(output)
                    return out
                return output
        except Exception:
            pass

    # PyTorch fallback
    y = _add_gelu_pytorch(input, other, alpha=alpha, approximate=approximate)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_add_gelu():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_gelu(input_tensor, other_tensor)

    # Test case 2: Test with alpha parameter
    alpha = 2
    results["test_case_2"] = add_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Test with approximate='tanh'
    approximate = 'tanh'
    results["test_case_3"] = add_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Test with a scalar 'other'
    other_scalar = 0.5
    results["test_case_4"] = add_gelu(input_tensor, other_scalar)

    return results

test_results = test_add_gelu()
