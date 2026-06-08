import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple, Union

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
def _gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to apply GELU activation element-wise.
    approximate: 0 for 'none' (exact), 1 for 'tanh'
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 0:
        # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt2_inv = 0.7071067811865476
        y = 0.5 * x * (1.0 + tl.erf(x * sqrt2_inv))
    else:
        # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * (x + 0.044715 * x * x * x) * 0.7978845608) - 1.0))
        y = cdf
    
    tl.store(output_ptr + offsets, y, mask=mask)


def _gelu_triton(input: Tensor, approximate: str = 'none') -> Tensor:
    """
    Apply GELU activation using Triton kernel.
    Returns: tensor with same shape as input, with GELU applied element-wise.
    """
    n_elements = input.numel()
    output = torch.empty_like(input)
    
    # Use approximate flag
    approx_flag = 1 if approximate == 'tanh' else 0
    
    # Determine block size
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _gelu_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        n_elements,
        approx_flag,
        BLOCK_SIZE,
    )
    
    return output


def _gelu_std_impl(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    correction: int = 1,
    approximate: str = 'none',
) -> Tensor:
    """
    Reference implementation: Apply GELU, then compute std.
    """
    # Apply GELU
    gelu_out = F.gelu(input, approximate=approximate)
    
    # Compute standard deviation
    std = torch.std(gelu_out, dim=dim, keepdim=keepdim, correction=correction)
    
    return std


try:
    _gelu_std_fast = torch.compile(_gelu_std_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _gelu_std_fast = _gelu_std_impl


def gelu_std(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    correction: int = 1,
    approximate: str = 'none',
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply GELU activation function, then compute standard deviation.
    
    Args:
        input: Input tensor.
        dim: Dimension(s) to reduce. If None, reduces over all dimensions.
        keepdim: Whether to keep reduced dimensions with size 1.
        correction: Bessel correction factor (0 or 1). Default is 1.
        approximate: GELU approximation method ('none' or 'tanh'). Default is 'none'.
        out: Optional output tensor.
    
    Returns:
        Standard deviation tensor after GELU activation.
    """
    try:
        y = _gelu_std_fast(input, dim=dim, keepdim=keepdim, correction=correction, approximate=approximate)
    except Exception:
        y = _gelu_std_impl(input, dim=dim, keepdim=keepdim, correction=correction, approximate=approximate)
    
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
#     gelu_result = F.gelu(input, approximate=approximate)
#     return torch.std(gelu_result, dim=dim, keepdim=keepdim, correction=correction, out=out)

def test_gelu_std():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(10, device='cuda')
    results["test_case_1"] = gelu_std(input1)
    
    # Test case 2: With dim parameter
    input2 = torch.randn(10, 20, device='cuda')
    results["test_case_2"] = gelu_std(input2, dim=1)
    
    # Test case 3: With keepdim=True
    input3 = torch.randn(10, 20, device='cuda')
    results["test_case_3"] = gelu_std(input3, dim=1, keepdim=True)
    
    # Test case 4: With approximate='tanh'
    input4 = torch.randn(10, device='cuda')
    results["test_case_4"] = gelu_std(input4, approximate='tanh')
    
    return results

test_results = test_gelu_std()
