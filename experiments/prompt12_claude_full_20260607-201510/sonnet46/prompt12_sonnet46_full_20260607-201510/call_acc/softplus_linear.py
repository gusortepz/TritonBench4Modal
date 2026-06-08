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
def _softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # softplus: if beta * x > threshold, return x (linear region)
    # else return (1/beta) * log(1 + exp(beta * x))
    beta_x = beta * x
    # Use log1p style: log(1 + exp(beta*x)) = log(exp(beta*x) * (exp(-beta*x) + 1))
    # For numerical stability in Triton, compute directly
    linear_out = x
    softplus_out = (1.0 / beta) * tl.log(1.0 + tl.exp(beta_x))
    # Select based on threshold
    result = tl.where(beta_x > threshold, linear_out, softplus_out)

    tl.store(out_ptr + offsets, result, mask=mask)


def _softplus_triton(x: Tensor, beta: float, threshold: float) -> Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _softplus_kernel[grid](
        x,
        out,
        n,
        beta=beta,
        threshold=threshold,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _softplus_pytorch(x: Tensor, beta: float, threshold: float) -> Tensor:
    return F.softplus(x, beta=beta, threshold=threshold)


def softplus_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    beta: float = 1,
    threshold: float = 20,
) -> Tensor:
    # Step 1: linear transformation
    y = F.linear(input, weight, bias)

    # Step 2: softplus activation
    if y.is_cuda and y.is_floating_point() and not y.is_complex():
        try:
            return _softplus_triton(y, float(beta), float(threshold))
        except Exception:
            return _softplus_pytorch(y, float(beta), float(threshold))
    else:
        return _softplus_pytorch(y, float(beta), float(threshold))

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
