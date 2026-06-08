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
def _mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    other_is_scalar,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if other_is_scalar:
        y = x * scalar_val
    else:
        o = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        y = x * o

    result = tl.maximum(y, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_relu_inplace_kernel(
    input_ptr,
    other_ptr,
    n_elements,
    other_is_scalar,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if other_is_scalar:
        y = x * scalar_val
    else:
        o = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        y = x * o

    result = tl.maximum(y, 0.0)
    tl.store(input_ptr + offsets, result, mask=mask)


def _mul_relu_pytorch(input: Tensor, other, inplace: bool = False) -> Tensor:
    if inplace and isinstance(other, Tensor):
        input.mul_(other)
        input.relu_()
        return input
    else:
        result = input * other
        return F.relu(result)


def mul_relu(input: Tensor, other, inplace: bool = False, out: Optional[Tensor] = None) -> Tensor:
    # Use Triton only for CUDA float tensors with compatible conditions
    is_scalar = isinstance(other, (int, float))
    other_is_tensor = isinstance(other, Tensor)

    use_triton = (
        input.is_cuda
        and input.is_contiguous()
        and input.dtype in (torch.float16, torch.float32, torch.float64)
        and (
            is_scalar
            or (
                other_is_tensor
                and other.is_cuda
                and other.is_contiguous()
                and other.dtype == input.dtype
                and other.shape == input.shape
            )
        )
    )

    if not use_triton:
        result = _mul_relu_pytorch(input, other, inplace=inplace)
        if out is not None:
            out.copy_(result)
            return out
        return result

    n_elements = input.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    scalar_val = float(other) if is_scalar else 0.0
    other_ptr = input if is_scalar else other  # dummy for scalar case

    if inplace and out is None:
        if is_scalar:
            _mul_relu_inplace_kernel[grid](
                input,
                input,  # dummy
                n_elements,
                True,
                scalar_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _mul_relu_inplace_kernel[grid](
                input,
                other,
                n_elements,
                False,
                0.0,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return input
    else:
        if out is not None:
            output = out
        else:
            output = torch.empty_like(input)

        if is_scalar:
            _mul_relu_kernel[grid](
                input,
                input,  # dummy
                output,
                n_elements,
                True,
                scalar_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _mul_relu_kernel[grid](
                input,
                other,
                output,
                n_elements,
                False,
                0.0,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def mul_relu(input, other, inplace=False, out=None):
#     result = torch.mul(input, other)
#     return F.relu(result, inplace=inplace)

def test_mul_relu():
    results = {}

    # Test case 1: Basic multiplication and ReLU with two tensors
    input1 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other1 = torch.tensor([1.0, -1.0, 1.0, -1.0], device='cuda')
    results["test_case_1"] = mul_relu(input1, other1)

    # Test case 2: Multiplication with a scalar
    input2 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = mul_relu(input2, other2)

    # Test case 3: In-place operation
    input3 = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
    other3 = torch.tensor([1.0, -1.0, 1.0, -1.0], device='cuda')
    results["test_case_3"] = mul_relu(input3, other3, inplace=True)

    # Test case 4: Multiplication with a different shaped tensor
    input4 = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], device='cuda')
    other4 = torch.tensor([[1.0, -1.0], [1.0, -1.0]], device='cuda')
    results["test_case_4"] = mul_relu(input4, other4)

    return results

test_results = test_mul_relu()
