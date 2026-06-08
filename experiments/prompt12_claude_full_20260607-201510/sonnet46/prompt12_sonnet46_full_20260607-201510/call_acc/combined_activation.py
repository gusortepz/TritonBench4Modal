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
def _sigmoid_tanh_mul_add_kernel(
    x_ptr,
    w2_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    w2_numel,
    bias_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # sigmoid(x)
    sig = tl.sigmoid(x)
    # tanh(x) = 2*sigmoid(2*x) - 1
    tanh_x = 2.0 * tl.sigmoid(2.0 * x) - 1.0

    # weight2: broadcast by cycling if needed
    w2_offsets = offsets % w2_numel
    w2 = tl.load(w2_ptr + w2_offsets, mask=mask, other=1.0)

    # bias: broadcast by cycling if needed
    bias_offsets = offsets % bias_numel
    b = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)

    result = sig * tanh_x * w2 + b
    tl.store(out_ptr + offsets, result, mask=mask)


def _sigmoid_tanh_mul_add_triton(x: Tensor, weight2: Tensor, bias: Tensor) -> Tensor:
    """Apply sigmoid(x) * tanh(x) * weight2 + bias using Triton."""
    out_shape = x.shape
    # Broadcast weight2 and bias to x shape
    w2 = weight2.expand(out_shape).contiguous()
    b = bias.expand(out_shape).contiguous()

    x_flat = x.contiguous().view(-1)
    w2_flat = w2.view(-1)
    b_flat = b.view(-1)

    n_elements = x_flat.numel()
    out_flat = torch.empty(n_elements, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(max(n_elements, 1)))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _sigmoid_tanh_mul_add_kernel[grid](
        x_flat,
        w2_flat,
        b_flat,
        out_flat,
        n_elements,
        w2_flat.numel(),
        b_flat.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.view(out_shape)


def _sigmoid_tanh_mul_add_pytorch(x: Tensor, weight2: Tensor, bias: Tensor) -> Tensor:
    sig = torch.sigmoid(x)
    tanh_x = torch.tanh(x)
    return sig * tanh_x * weight2 + bias


def combined_activation(
    input: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Step 1: Matrix multiplication over last dim
    # input: (*, N, D_in), weight1: (D_in, D_out) -> x: (*, N, D_out)
    x = torch.matmul(input, weight1)

    # Step 2: Apply sigmoid(x) * tanh(x) * weight2 + bias
    if (
        x.is_cuda
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and weight2 is not None
        and bias is not None
    ):
        try:
            # Ensure weight2 and bias are broadcastable
            w2 = weight2.to(dtype=x.dtype, device=x.device)
            b = bias.to(dtype=x.dtype, device=x.device)
            y = _sigmoid_tanh_mul_add_triton(x, w2, b)
        except Exception:
            y = _sigmoid_tanh_mul_add_pytorch(x, weight2, bias)
    else:
        y = _sigmoid_tanh_mul_add_pytorch(x, weight2, bias)

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
