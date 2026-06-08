import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Union
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
def _sub_gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Sub: x - alpha * other
    result = x - alpha * other_val
    
    # GELU
    if approximate == "tanh":
        # GELU (tanh approximation):
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        SQRT_2_OVER_PI = 0.7978845608028654
        COEFF = 0.044715
        x_cubed = result * result * result
        arg = SQRT_2_OVER_PI * (result + COEFF * x_cubed)
        tanh_val = 2.0 * tl.sigmoid(2.0 * arg) - 1.0
        gelu_result = 0.5 * result * (1.0 + tanh_val)
    else:
        # GELU (exact):
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_result = 0.5 * result * (1.0 + tl.erf(result * 0.7071067811865476))
    
    tl.store(output_ptr + offsets, gelu_result, mask=mask)


def sub_gelu(
    input: Tensor,
    other: Union[Tensor, float],
    alpha: Union[int, float] = 1,
    approximate: str = "none",
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Subtracts 'other', scaled by 'alpha', from 'input', and applies GELU.
    
    Args:
        input: The input tensor.
        other: The tensor or number to subtract from input.
        alpha: The multiplier for other. Default is 1.
        approximate: The approximation method for GELU ('none' or 'tanh'). Default is 'none'.
        out: The output tensor (optional).
    
    Returns:
        The result tensor after subtraction and GELU activation.
    """
    # Validate input
    if not isinstance(input, Tensor):
        raise TypeError(f"input must be a Tensor, got {type(input)}")
    
    # Convert other to tensor if needed
    if isinstance(other, (int, float)):
        other_tensor = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other_tensor = other
    
    # Ensure tensors are on the same device and dtype
    if input.device != other_tensor.device:
        other_tensor = other_tensor.to(device=input.device)
    if input.dtype != other_tensor.dtype:
        other_tensor = other_tensor.to(dtype=input.dtype)
    
    # Broadcast other to match input shape if necessary
    if other_tensor.shape != input.shape:
        other_tensor = torch.broadcast_to(other_tensor, input.shape)
    
    # Use Triton only for CUDA float32/float64 tensors
    use_triton = (
        input.is_cuda
        and input.dtype in (torch.float32, torch.float64)
        and other_tensor.is_cuda
        and other_tensor.dtype in (torch.float32, torch.float64)
    )
    
    if use_triton:
        try:
            n_elements = input.numel()
            output = torch.empty_like(input)
            
            # Determine block size
            BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
            
            # Map approximate string to constexpr value
            approx_const = "tanh" if approximate == "tanh" else "none"
            
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            _sub_gelu_kernel[grid](
                input,
                other_tensor,
                output,
                n_elements,
                alpha=alpha,
                approximate=approx_const,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            if out is not None:
                out.copy_(output)
                return out
            return output
        except Exception:
            # Fallback to PyTorch
            pass
    
    # PyTorch reference implementation
    y = input - alpha * other_tensor
    
    if approximate == "tanh":
        y = F.gelu(y, approximate="tanh")
    else:
        y = F.gelu(y, approximate="none")
    
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_sub_gelu():
    results = {}

    # Test case 1: Basic subtraction and GELU with default approximate
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 1.0, 1.5], device='cuda')
    results["test_case_1"] = sub_gelu(input_tensor, other_tensor)

    # Test case 2: Subtraction with alpha and GELU with default approximate
    alpha = 0.5
    results["test_case_2"] = sub_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Subtraction and GELU with 'tanh' approximation
    approximate = 'tanh'
    results["test_case_3"] = sub_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Subtraction with alpha and GELU with 'tanh' approximation
    results["test_case_4"] = sub_gelu(input_tensor, other_tensor, alpha=alpha, approximate=approximate)

    return results

test_results = test_sub_gelu()
