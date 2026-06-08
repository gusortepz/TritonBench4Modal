import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple
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
    cos_ptr,
    signbit_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    cos_val = tl.cos(x)
    signbit_val = tl.where(cos_val < 0.0, 1, 0)
    
    tl.store(cos_ptr + offsets, cos_val, mask=mask)
    tl.store(signbit_ptr + offsets, signbit_val, mask=mask)


def cos_signbit(input: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the cosine of each element in the input tensor and the sign bit of the cosine result.
    
    Args:
        input (Tensor): The input tensor for which the cosine and sign bit are computed.
    
    Returns:
        Tuple[Tensor, Tensor]: A tuple of (cos_output, signbit_output) where:
            - cos_output: Tensor of same shape and dtype as input, containing the cosine values.
            - signbit_output: Tensor of same shape as input with dtype int32, containing sign bits (0 if cos >= 0, 1 if cos < 0).
    """
    if not input.is_cuda or input.dtype not in [torch.float32, torch.float64]:
        cos_val = torch.cos(input)
        signbit_val = torch.where(cos_val < 0, 1, 0).to(torch.int32)
        return cos_val, signbit_val
    
    n_elements = input.numel()
    cos_output = torch.empty_like(input)
    signbit_output = torch.zeros(input.shape, dtype=torch.int32, device=input.device)
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _cos_signbit_kernel[grid](
        input,
        cos_output,
        signbit_output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_output, signbit_output

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
