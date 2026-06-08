import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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
def _selu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # SELU constants
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    
    # SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(0.0, alpha * (tl.exp(x) - 1.0))
    y = scale * (pos + neg)
    
    tl.store(output_ptr + offsets, y, mask=mask)


def _selu_triton(input: Tensor) -> Tensor:
    """Triton implementation of SELU."""
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        return F.selu(input)
    
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    if n_elements == 0:
        return output
    
    # Use fixed BLOCK_SIZE with cap to avoid OOM on large inputs
    BLOCK_SIZE = min(triton.next_power_of_2(min(n_elements, 65536)), 1024)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _selu_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Applies the element-wise SELU (Scaled Exponential Linear Unit) function.
    
    SELU is defined as: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    where alpha ≈ 1.673 and scale ≈ 1.051.
    
    Args:
        input: input tensor
        inplace: if True, modifies input in-place
    
    Returns:
        output tensor with SELU applied
    """
    if input.is_cuda and input.dtype in (torch.float32, torch.float64):
        try:
            y = _selu_triton(input)
        except Exception:
            y = F.selu(input)
    else:
        y = F.selu(input)
    
    if inplace:
        input.copy_(y)
        return input
    return y

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
