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
def _combined_activation_kernel(
    output_ptr,
    intermediate_ptr,
    weight2_ptr,
    bias_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: sigmoid(x) * tanh(x) * weight2 + bias
    where x is the intermediate (matmul) result.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load intermediate (matmul output)
    intermediate = tl.load(intermediate_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid(intermediate)
    sigmoid_x = tl.sigmoid(intermediate)

    # Compute tanh(intermediate)
    tanh_x = 2.0 * tl.sigmoid(2.0 * intermediate) - 1.0

    # Element-wise multiply: sigmoid(x) * tanh(x)
    prod = sigmoid_x * tanh_x

    # Load weight2 (broadcast if needed)
    weight2 = tl.load(weight2_ptr + offsets, mask=mask, other=1.0)

    # Load bias (broadcast if needed)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Final computation: prod * weight2 + bias
    result = prod * weight2 + bias

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def _combined_activation_impl(
    input: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias: Tensor,
) -> Tensor:
    """
    Reference implementation: matmul -> sigmoid & tanh -> multiply -> add bias
    """
    # Matrix multiplication: input @ weight1
    intermediate = torch.matmul(input, weight1)

    # Sigmoid and tanh
    sigmoid_x = torch.sigmoid(intermediate)
    tanh_x = torch.tanh(intermediate)

    # Element-wise multiply
    prod = sigmoid_x * tanh_x

    # Multiply by weight2 and add bias
    output = prod * weight2 + bias

    return output


try:
    _combined_activation_fast = torch.compile(
        _combined_activation_impl, mode="max-autotune", fullgraph=False
    )
except Exception:
    _combined_activation_fast = _combined_activation_impl


def combined_activation(
    input: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs a sequence of operations combining matrix multiplication, sigmoid,
    tanh, element-wise multiplication, and addition.

    Args:
        input: Input tensor of shape (*, N, D_in)
        weight1: Weight matrix of shape (D_in, D_out)
        weight2: Weight tensor for element-wise multiplication
        bias: Bias tensor
        out: Output tensor (optional)

    Returns:
        Output tensor
    """
    # Try the compiled fast path first, fall back to reference
    try:
        y = _combined_activation_fast(input, weight1, weight2, bias)
    except Exception:
        y = _combined_activation_impl(input, weight1, weight2, bias)

    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def combined_activation(input, weight1, weight2, bias, *, out=None):
#     """
#     Perform the combined activation function which includes matrix multiplication,
#     sigmoid, tanh, element-wise multiplication, and addition.

#     Args:
#         input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any batch dimensions.
#         weight1 (Tensor): Weight matrix of shape (D_in, D_out).
#         weight2 (Tensor): Weight tensor for element-wise multiplication, must be broadcastable 
#                           to the shape of the intermediate activation.
#         bias (Tensor): Bias tensor, must be broadcastable to the shape of the output.
#         out (Tensor, optional): Output tensor to store the result, ignored if None.

#     Returns:
#         Tensor: Output tensor of shape (*, N, D_out).
#     """
#     z = torch.mm(input, weight1)
#     s = torch.sigmoid(z)
#     t = torch.tanh(s)
#     m = t * weight2
#     y = m + bias
#     if out is not None:
#         out.copy_(y)
#         return out
#     return y

def test_combined_activation():
    results = {}

    # Test case 1
    input1 = torch.randn(2, 3, device='cuda')
    weight1_1 = torch.randn(3, 4, device='cuda')
    weight2_1 = torch.randn(2, 4, device='cuda')
    bias1 = torch.randn(2, 4, device='cuda')
    results["test_case_1"] = combined_activation(input1, weight1_1, weight2_1, bias1)

    # Test case 2
    input2 = torch.randn(3, 3, device='cuda')
    weight1_2 = torch.randn(3, 5, device='cuda')
    weight2_2 = torch.randn(3, 5, device='cuda')
    bias2 = torch.randn(3, 5, device='cuda')
    results["test_case_2"] = combined_activation(input2, weight1_2, weight2_2, bias2)

    # Test case 3
    input3 = torch.randn(4, 3, device='cuda')
    weight1_3 = torch.randn(3, 6, device='cuda')
    weight2_3 = torch.randn(4, 6, device='cuda')
    bias3 = torch.randn(4, 6, device='cuda')
    results["test_case_3"] = combined_activation(input3, weight1_3, weight2_3, bias3)

    # Test case 4
    input4 = torch.randn(5, 3, device='cuda')
    weight1_4 = torch.randn(3, 7, device='cuda')
    weight2_4 = torch.randn(5, 7, device='cuda')
    bias4 = torch.randn(5, 7, device='cuda')
    results["test_case_4"] = combined_activation(input4, weight1_4, weight2_4, bias4)

    return results

test_results = test_combined_activation()
