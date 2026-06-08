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
def _gelu_kernel(
    input_ptr,
    output_ptr,
    numel,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 1:
        # GELU tanh approximation
        cdf = 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        y = x * cdf
    else:
        # GELU exact
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    
    tl.store(output_ptr + offsets, y, mask=mask)


def _gelu_triton(input: Tensor, approximate: str = 'none') -> Tensor:
    """Apply GELU activation using Triton kernel."""
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return F.gelu(input, approximate=approximate)
    
    output = torch.empty_like(input)
    numel = input.numel()
    
    if numel == 0:
        return output
    
    approximate_int = 1 if approximate == 'tanh' else 0
    BLOCK_SIZE = min(triton.next_power_of_2(numel), 1024)
    grid = lambda meta: (triton.cdiv(numel, BLOCK_SIZE),)
    
    _gelu_kernel[grid](
        input,
        output,
        numel,
        approximate_int,
        BLOCK_SIZE,
    )
    
    return output


def min_gelu(
    input: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    approximate: str = 'none',
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes GELU activation then returns the minimum value along dimension(s).
    
    Args:
        input: Input tensor
        dim: Dimension to reduce over. If None, reduces all dimensions.
        keepdim: Whether to retain the reduced dimension as size 1.
        approximate: GELU approximation method ('none' for exact, 'tanh' for approximate).
        out: Optional output tensor.
    
    Returns:
        Tensor of minimum values after GELU activation.
    """
    if input.is_cuda and input.dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            gelu_out = _gelu_triton(input, approximate=approximate)
        except Exception:
            gelu_out = F.gelu(input, approximate=approximate)
    else:
        gelu_out = F.gelu(input, approximate=approximate)
    
    if dim is None:
        y = torch.min(gelu_out)
    else:
        y = torch.min(gelu_out, dim=dim, keepdim=keepdim).values
    
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def min_gelu(input: Tensor, dim=None, keepdim=False, approximate='none', out=None) -> Tensor:
#     """
#     Computes the minimum of the GELU activation of the input tensor along the specified dimension(s).
    
#     Args:
#         input (Tensor): The input tensor.
#         dim (int, optional): The dimension to reduce. If None, returns the minimum of all elements.
#         keepdim (bool, optional): Whether the output tensor retains :attr:`dim` as size 1. Default is False.
#         approximate (str, optional): The approximation method for GELU. Default is 'none'.
#                                       'none' computes exact GELU, 'tanh' computes the approximate GELU using the tanh method.
#         out (Tensor, optional): The output tensor.

#     Returns:
#         Tensor: The minimum value after applying GELU.
#         If dim is specified, returns a namedtuple (values, indices), otherwise returns the minimum value tensor.
#     """
#     if approximate == 'none':
#         gelu_input = input * torch.erf(input / torch.sqrt(torch.tensor(2.0))) / 2.0
#     elif approximate == 'tanh':
#         gelu_input = 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (input + 0.044715 * input ** 3)))
#     else:
#         raise ValueError(f"Invalid value for approximate: {approximate}. Choose 'none' or 'tanh'.")
#     if dim is not None:
#         return torch.min(gelu_input, dim=dim, keepdim=keepdim, out=out)
#     else:
#         return torch.min(gelu_input, out=out)

def test_min_gelu():
    results = {}
    
    # Test case 1: Default parameters
    input_tensor = torch.tensor([1.0, -0.5, 0.0, 2.0], device='cuda')
    results["test_case_1"] = min_gelu(input_tensor)
    
    # Test case 2: With dimension reduction
    input_tensor = torch.tensor([[1.0, -0.5], [0.0, 2.0]], device='cuda')
    results["test_case_2"] = min_gelu(input_tensor, dim=1)
    
    # Test case 3: With dimension reduction and keepdim=True
    input_tensor = torch.tensor([[1.0, -0.5], [0.0, 2.0]], device='cuda')
    results["test_case_3"] = min_gelu(input_tensor, dim=1, keepdim=True)
    
    # Test case 4: Using 'tanh' approximation
    input_tensor = torch.tensor([1.0, -0.5, 0.0, 2.0], device='cuda')
    results["test_case_4"] = min_gelu(input_tensor, approximate='tanh')
    
    return results

test_results = test_min_gelu()
