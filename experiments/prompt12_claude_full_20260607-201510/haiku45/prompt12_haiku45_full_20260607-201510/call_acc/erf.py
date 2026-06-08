import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

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
def _erf_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.math.erf(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def erf(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        y = torch.erf(input)
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _erf_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    if out is not None:
        out.copy_(output)
        return out
    return output

##################################################################################################################################################



import torch

# def erf(input_tensor):
#     """
#     计算输入张量的误差函数（error function）。

#     参数：
#     input_tensor (Tensor): 输入的张量。

#     返回：
#     Tensor: 输入张量中每个元素的误差函数值。
#     """
#     return torch.special.erf(input_tensor)

def test_erf():
    results = {}
    
    # Test case 1: Single element tensor
    input_tensor = torch.tensor([0.5], device='cuda')
    results["test_case_1"] = erf(input_tensor)
    
    # Test case 2: Multi-element tensor
    input_tensor = torch.tensor([0.5, -1.0, 2.0], device='cuda')
    results["test_case_2"] = erf(input_tensor)
    
    # Test case 3: Large values tensor
    input_tensor = torch.tensor([10.0, -10.0], device='cuda')
    results["test_case_3"] = erf(input_tensor)
    
    # Test case 4: Zero tensor
    input_tensor = torch.tensor([0.0], device='cuda')
    results["test_case_4"] = erf(input_tensor)
    
    return results

test_results = test_erf()
