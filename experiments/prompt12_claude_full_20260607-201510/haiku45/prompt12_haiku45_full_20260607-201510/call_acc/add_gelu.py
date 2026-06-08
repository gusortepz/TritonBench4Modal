import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

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
def _add_gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    numel,
    alpha: tl.constexpr,
    is_other_scalar: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if is_other_scalar:
        other_val = tl.load(other_ptr)
        y = x + alpha * other_val
    else:
        other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        y = x + alpha * other_val

    if approximate == "tanh":
        cdf = 0.5 * (
            1.0
            + (
                2.0
                * tl.sigmoid(
                    2.0 * (y + 0.044715 * y * y * y) * 0.7978845608028654
                )
                - 1.0
            )
        )
        result = y * cdf
    else:
        result = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))

    tl.store(output_ptr + offsets, result, mask=mask)


def add_gelu(
    input: Tensor,
    other: Union[Tensor, float, int],
    alpha: Union[float, int] = 1,
    approximate: str = "none",
    out: Optional[Tensor] = None,
) -> Tensor:
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        y = input + alpha * other
        if approximate == "tanh":
            y = F.gelu(y, approximate="tanh")
        else:
            y = F.gelu(y, approximate="none")
        if out is not None:
            out.copy_(y)
            return out
        return y

    if isinstance(other, (int, float)):
        other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
        is_other_scalar = True
    else:
        other_tensor = other
        is_other_scalar = False

    numel = input.numel()
    output = torch.empty_like(input)

    BLOCK_SIZE = min(triton.next_power_of_2(numel), 1024)
    grid = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _add_gelu_kernel[grid](
        input,
        other_tensor,
        output,
        numel,
        alpha=alpha,
        is_other_scalar=is_other_scalar,
        approximate=approximate,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if out is not None:
        out.copy_(output)
        return out
    return output

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
