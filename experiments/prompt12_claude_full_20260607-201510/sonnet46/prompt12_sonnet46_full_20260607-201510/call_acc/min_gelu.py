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
def _gelu_exact_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
    result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _gelu_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tanh approximation gelu
    inner = (x + 0.044715 * x * x * x) * 0.7978845608028654
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, result, mask=mask)


def _apply_gelu_triton(input: Tensor, approximate: str) -> Tensor:
    """Apply GELU using Triton kernel for CUDA float tensors."""
    x = input.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n)) if n > 0 else 1
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if approximate == 'tanh':
        _gelu_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        _gelu_exact_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _apply_gelu_pytorch(input: Tensor, approximate: str) -> Tensor:
    """Apply GELU using PyTorch."""
    return F.gelu(input, approximate=approximate)


def min_gelu(input: Tensor, dim=None, keepdim: bool = False, approximate: str = 'none', out: Optional[Tensor] = None) -> Tensor:
    """
    Computes GELU activation on input tensor, then returns the minimum value
    along the specified dimension(s) or over all elements if no dimension is specified.
    
    Args:
        input: The input tensor.
        dim: The dimension to reduce. If None, returns the minimum of all elements.
        keepdim: Whether the output tensor retains dim as size 1.
        approximate: The approximation method for GELU ('none' or 'tanh').
        out: The output tensor.
    
    Returns:
        Tensor with minimum of GELU-activated input.
    """
    # Validate approximate parameter
    if approximate not in ('none', 'tanh'):
        raise ValueError(f"approximate must be 'none' or 'tanh', got '{approximate}'")
    
    # Apply GELU
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        try:
            gelu_out = _apply_gelu_triton(input, approximate)
        except Exception:
            gelu_out = _apply_gelu_pytorch(input, approximate)
    else:
        gelu_out = _apply_gelu_pytorch(input, approximate)
    
    # Compute min
    if dim is None:
        # Return minimum of all elements
        if out is not None:
            y = torch.min(gelu_out)
            out.copy_(y)
            return out
        return torch.min(gelu_out)
    else:
        # Return minimum along specified dimension
        # torch.min with dim returns (values, indices), we want values
        result = torch.min(gelu_out, dim=dim, keepdim=keepdim)
        y = result.values
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
