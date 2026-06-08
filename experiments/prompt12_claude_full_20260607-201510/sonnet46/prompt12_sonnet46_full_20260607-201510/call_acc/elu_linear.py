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
def _elu_kernel(
    x_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ELU: x if x > 0 else alpha * (exp(x) - 1)
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(x, 0.0)
    elu_neg = alpha * (tl.exp(neg) - 1.0)
    # combine: pos part + elu_neg part
    # when x > 0: pos = x, elu_neg = alpha*(exp(0)-1) = 0
    # when x <= 0: pos = 0, elu_neg = alpha*(exp(x)-1)
    result = pos + elu_neg
    tl.store(out_ptr + offsets, result, mask=mask)


def _elu_triton(x: Tensor, alpha: float) -> Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n, 1)))
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _elu_kernel[grid](
        x,
        out,
        alpha,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _elu_linear_impl(input: Tensor, weight: Tensor, bias: Optional[Tensor], alpha: float) -> Tensor:
    # Linear transformation
    y = F.linear(input, weight, bias)
    # ELU activation
    if y.is_cuda and y.is_contiguous() and y.dtype in (torch.float16, torch.float32, torch.bfloat16):
        y_cont = y.contiguous().float() if y.dtype == torch.float16 else y.contiguous()
        try:
            result = _elu_triton(y_cont, alpha)
            if y.dtype != y_cont.dtype:
                result = result.to(y.dtype)
            return result
        except Exception:
            pass
    return F.elu(y, alpha=alpha)


def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False) -> Tensor:
    """
    Applies a linear transformation to the input tensor, followed by the ELU activation.
    
    Args:
        input (Tensor): The input tensor for the linear layer.
        weight (Tensor): The weight tensor for the linear transformation.
        bias (Tensor, optional): The bias tensor. Default: None.
        alpha (float, optional): The alpha parameter for ELU. Default: 1.0.
        inplace (bool, optional): Whether to apply ELU in-place. Default: False.
    
    Returns:
        Tensor: Output after linear transformation and ELU activation.
    """
    # Linear transformation
    y = F.linear(input, weight, bias)
    
    # ELU activation
    if (y.is_cuda and 
        y.is_contiguous() and 
        y.dtype == torch.float32 and
        not inplace):
        try:
            result = _elu_triton(y, alpha)
            return result
        except Exception:
            pass
    
    # Fallback to PyTorch ELU
    return F.elu(y, alpha=alpha, inplace=inplace)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
#     output = F.linear(input, weight, bias)
#     return F.elu(output, alpha=alpha, inplace=inplace)

def test_elu_linear():
    results = {}

    # Test case 1: Basic test with bias, alpha=1.0, inplace=False
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = elu_linear(input1, weight1, bias1)

    # Test case 2: Without bias, alpha=1.0, inplace=False
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    results["test_case_2"] = elu_linear(input2, weight2)

    # Test case 3: With bias, alpha=0.5, inplace=False
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight3 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = elu_linear(input3, weight3, bias3, alpha=0.5)

    # Test case 4: With bias, alpha=1.0, inplace=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias4 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_4"] = elu_linear(input4, weight4, bias4, inplace=True)

    return results

test_results = test_elu_linear()
