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
def _relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    relu_val = tl.maximum(input_val, 0.0)

    result = tl.sqrt(relu_val)

    tl.store(output_ptr + offsets, result, mask=mask)


def relu_sqrt(input: Tensor, inplace: bool = False, out: Optional[Tensor] = None) -> Tensor:
    if input.numel() == 0:
        if out is not None:
            return out
        return input.clone()

    if out is not None:
        output = out
    elif inplace:
        output = input
    else:
        output = torch.empty_like(input)

    n_elements = input.numel()
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024 * 64)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _relu_sqrt_kernel[grid](input, output, n_elements, BLOCK_SIZE)

    return output

##################################################################################################################################################



import torch
from torch import Tensor

# def relu_sqrt(input: Tensor, inplace: bool=False, out: Tensor=None) -> Tensor:
#     if input.dtype != torch.float32 and input.dtype != torch.float64:
#         input = input.float()
#     if inplace:
#         input.relu_()
#         input.sqrt_()
#         return input
#     elif out is not None:
#         out.copy_(torch.sqrt(torch.relu(input)))
#         return out
#     else:
#         return torch.sqrt(torch.relu(input))

def test_relu_sqrt():
    results = {}
    
    # Test case 1: Default parameters
    a = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = relu_sqrt(a)
    
    # Test case 2: Inplace operation
    b = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_2"] = relu_sqrt(b, inplace=True)
    
    # Test case 3: Out parameter
    c = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    out = torch.empty_like(c)
    results["test_case_3"] = relu_sqrt(c, out=out)
    
    # Test case 4: Non-float input
    d = torch.tensor([-1, 0, 4, 9], device='cuda')
    results["test_case_4"] = relu_sqrt(d)
    
    return results

test_results = test_relu_sqrt()
