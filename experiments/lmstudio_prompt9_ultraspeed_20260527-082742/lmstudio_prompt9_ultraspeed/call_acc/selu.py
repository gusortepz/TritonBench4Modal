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
def _selu_kernel(x_ptr, out_ptr, n, ALPHA: tl.constexpr, SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = SCALE * (tl.maximum(x, 0.0) + tl.minimum(0.0, ALPHA * (tl.exp(x) - 1.0)))
    tl.store(out_ptr + offs, y, mask=mask)

def selu(input, inplace=False):
    if input.is_cuda and input.is_floating_point():
        x_c = input.contiguous()
        n = x_c.numel()
        out = torch.empty_like(x_c)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _selu_kernel[grid](x_c, out, n, ALPHA=1.6732632423543772, SCALE=1.0507009873554805, BLOCK_SIZE=1024)
        if inplace:
            input.copy_(out)
            return input
        return out
    return F.selu(input, inplace=inplace)

##################################################################################################################################################



def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    return results

test_results = test_selu()
