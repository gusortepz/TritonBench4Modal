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
def _softplus_linear_kernel(
    out_ptr,
    input_ptr,
    linear_ptr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel: applies softplus(beta * x, threshold) element-wise.
    Assumes linear transformation already applied to compute linear_ptr.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(linear_ptr + offsets, mask=mask, other=0.0)
    
    # Softplus: log(1 + exp(beta * x)) for x < threshold, else beta * x
    scaled = beta * x
    condition = scaled < threshold
    result = tl.where(
        condition,
        (1.0 / beta) * tl.log(1.0 + tl.exp(scaled)),
        x
    )
    
    tl.store(out_ptr + offsets, result, mask=mask)


def softplus_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tensor:
    """
    Applies a linear transformation followed by Softplus activation.
    
    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias vector of shape (out_features,)
        beta: Scaling parameter for softplus (default: 1)
        threshold: Threshold above which softplus reverts to linear (default: 20)
    
    Returns:
        Output tensor of shape (..., out_features)
    """
    # PyTorch reference path: linear + softplus
    linear_out = F.linear(input, weight, bias)
    
    # Check if Triton can be used
    if not (input.is_cuda and linear_out.dtype in [torch.float32, torch.float64]):
        # Fall back to PyTorch for CPU or unsupported dtypes
        return F.softplus(linear_out, beta=beta, threshold=threshold)
    
    # Triton path for CUDA float32/float64
    output = torch.empty_like(linear_out)
    n_elements = linear_out.numel()
    
    if n_elements == 0:
        return output
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    
    _softplus_linear_kernel[grid](
        output,
        input,
        linear_out,
        beta=beta,
        threshold=threshold,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

##################################################################################################################################################



import torch

def test_softplus_linear():
    results = {}

    # Test case 1: Basic test with default beta and threshold
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = softplus_linear(input1, weight1, bias1)

    # Test case 2: Test with non-default beta
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias2 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_2"] = softplus_linear(input2, weight2, bias2, beta=2)

    # Test case 3: Test with non-default threshold
    input3 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device='cuda')
    weight3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = softplus_linear(input3, weight3, bias3, threshold=15)

    # Test case 4: Test with no bias
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = softplus_linear(input4, weight4)

    return results

test_results = test_softplus_linear()
