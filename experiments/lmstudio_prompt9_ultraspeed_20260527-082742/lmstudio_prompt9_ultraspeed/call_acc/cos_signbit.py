import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@triton.jit
def _cos_signbit_kernel(x_ptr, cos_ptr, sign_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = tl.cos(x)
    s = (c < 0).to(tl.int32)
    tl.store(cos_ptr + offs, c, mask=mask)
    tl.store(sign_ptr + offs, s, mask=mask)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not input.is_cuda or not input.is_floating_point():
        c = torch.cos(input)
        s = (c < 0).to(torch.int32)
        return c, s
    n = input.numel()
    cos_out = torch.empty_like(input)
    sign_out = torch.empty(n, dtype=torch.int32, device=input.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _cos_signbit_kernel[grid](input, cos_out, sign_out, n, BLOCK_SIZE=1024)
    return cos_out, sign_out

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
