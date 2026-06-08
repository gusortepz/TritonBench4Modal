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
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # sigmoid: 1 / (1 + exp(-x)) = tl.sigmoid(x)
    out = tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def _apply_sigmoid_triton(input: Tensor) -> Tensor:
    """Apply sigmoid using Triton kernel for CUDA float tensors."""
    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _sigmoid_kernel[grid](
        input.contiguous().view(-1),
        out.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def sigmoid_argmax(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Applies the sigmoid function to each element in the input and then
    computes the indices of the maximum values along the specified dimension
    or over all elements if no dimension is specified.

    Args:
        input (Tensor): The input tensor.
        dim (int, optional): The dimension to reduce. Default is None,
            which computes the argmax over all elements.
        keepdim (bool, optional): Whether the output tensor has dim retained
            or not. Default is False.

    Returns:
        LongTensor: The indices of the maximum values.
    """
    # Apply sigmoid: since sigmoid is monotonically increasing,
    # argmax(sigmoid(x)) == argmax(x), but we apply it faithfully as described.
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        try:
            sig = _apply_sigmoid_triton(input)
        except Exception:
            sig = torch.sigmoid(input)
    else:
        sig = torch.sigmoid(input)

    if dim is None:
        # Argmax over all elements (flattened)
        return torch.argmax(sig.reshape(-1))
    else:
        return torch.argmax(sig, dim=dim, keepdim=keepdim)

##################################################################################################################################################



import torch

def test_sigmoid_argmax():
    results = {}

    # Test case 1: 1D tensor, no dim specified
    input1 = torch.tensor([0.1, 2.0, -1.0, 3.0], device='cuda')
    results["test_case_1"] = sigmoid_argmax(input1)

    # Test case 2: 2D tensor, dim=0
    input2 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_2"] = sigmoid_argmax(input2, dim=0)

    # Test case 3: 2D tensor, dim=1
    input3 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_3"] = sigmoid_argmax(input3, dim=1)

    # Test case 4: 2D tensor, dim=1, keepdim=True
    input4 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_4"] = sigmoid_argmax(input4, dim=1, keepdim=True)

    return results

test_results = test_sigmoid_argmax()
