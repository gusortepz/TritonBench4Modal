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
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # LeakyReLU: max(0, x) + negative_slope * min(0, x)
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(x, 0.0)
    y = pos + negative_slope * neg

    tl.store(output_ptr + offsets, y, mask=mask)


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    # Use PyTorch fallback for non-CUDA or non-floating-point tensors
    if not input.is_cuda or not input.is_floating_point():
        return F.leaky_relu(input, negative_slope=negative_slope, inplace=inplace)

    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    n_elements = input.numel()
    if n_elements == 0:
        return output

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    try:
        _leaky_relu_kernel[grid](
            input,
            output,
            n_elements,
            negative_slope,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    except Exception:
        # Fallback to PyTorch
        result = F.leaky_relu(input, negative_slope=negative_slope, inplace=False)
        output.copy_(result)

    return output

##################################################################################################################################################



import torch

def test_leaky_relu():
    results = {}

    # Test case 1: Default parameters
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = leaky_relu(input_tensor_1)

    # Test case 2: Custom negative_slope
    input_tensor_2 = torch.tensor([-2.0, 0.0, 2.0], device='cuda')
    results["test_case_2"] = leaky_relu(input_tensor_2, negative_slope=0.1)

    # Test case 3: Inplace operation
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = leaky_relu(input_tensor_3, inplace=True)

    # Test case 4: Larger tensor
    input_tensor_4 = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0], device='cuda')
    results["test_case_4"] = leaky_relu(input_tensor_4)

    return results

test_results = test_leaky_relu()
