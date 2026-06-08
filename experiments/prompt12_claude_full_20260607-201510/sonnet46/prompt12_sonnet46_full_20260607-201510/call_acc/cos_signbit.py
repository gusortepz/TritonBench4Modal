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
def _cos_signbit_kernel(
    input_ptr,
    cos_out_ptr,
    sign_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.cos(x)
    # signbit: True if value < 0.0 (negative), False if >= 0.0 (positive)
    sign_val = cos_val < 0.0

    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sign_out_ptr + offsets, sign_val, mask=mask)


def _cos_signbit_pytorch(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_result = torch.cos(input)
    sign_result = torch.signbit(cos_result)
    return cos_result, sign_result


def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cosine of each element in the input tensor, followed by
    determining the sign bit for each cosine result, indicating if it is
    positive or negative.

    Args:
        input (Tensor): The input tensor for which the cosine and sign bit are computed.

    Returns:
        Tuple[Tensor, Tensor]: (cos_result, signbit_result)
    """
    # Use PyTorch fallback for non-CUDA or non-float inputs
    if not input.is_cuda or input.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return _cos_signbit_pytorch(input)

    try:
        # Work in float32 for Triton kernel (cos not available for all dtypes)
        original_dtype = input.dtype
        if input.dtype in (torch.float16, torch.bfloat16):
            x = input.float()
        else:
            x = input

        n_elements = x.numel()
        cos_out = torch.empty_like(x)
        sign_out = torch.empty(x.shape, dtype=torch.bool, device=x.device)

        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        if BLOCK_SIZE < 1:
            BLOCK_SIZE = 1
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _cos_signbit_kernel[grid](
            x,
            cos_out,
            sign_out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Cast cos output back to original dtype if needed
        if original_dtype in (torch.float16, torch.bfloat16):
            cos_out = cos_out.to(original_dtype)

        return cos_out, sign_out

    except Exception:
        return _cos_signbit_pytorch(input)

##################################################################################################################################################



import torch
from typing import Tuple

# def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     cos_result = torch.cos(input)
#     sign_bit = torch.signbit(cos_result)
#     return (cos_result, sign_bit)

def test_cos_signbit():
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    cos_result_1, sign_bit_1 = cos_signbit(input_tensor_1)
    results["test_case_1"] = (cos_result_1.cpu(), sign_bit_1.cpu())

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    cos_result_2, sign_bit_2 = cos_signbit(input_tensor_2)
    results["test_case_2"] = (cos_result_2.cpu(), sign_bit_2.cpu())

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    cos_result_3, sign_bit_3 = cos_signbit(input_tensor_3)
    results["test_case_3"] = (cos_result_3.cpu(), sign_bit_3.cpu())

    # Test case 4: Edge case with pi multiples
    input_tensor_4 = torch.tensor([torch.pi, -torch.pi, 2*torch.pi], device='cuda')
    cos_result_4, sign_bit_4 = cos_signbit(input_tensor_4)
    results["test_case_4"] = (cos_result_4.cpu(), sign_bit_4.cpu())

    return results

test_results = test_cos_signbit()
