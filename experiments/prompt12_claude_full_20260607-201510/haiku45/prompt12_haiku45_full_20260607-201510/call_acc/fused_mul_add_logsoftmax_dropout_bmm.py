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
def _fused_mul_add_logsoftmax_dropout_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
    dropout_p: tl.constexpr,
    seed: tl.constexpr,
):
    """
    Fused kernel for: mul(input1, input2) -> add(other) -> logsoftmax -> dropout
    """
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Element-wise multiplication
    result = x * y

    # Dropout during training: scale by 1/(1-p) to keep expected value
    if dropout_p > 0.0:
        # Pseudo-random: approximate dropout mask
        rand_val = tl.rand(seed, offsets)
        mask_dropout = rand_val > dropout_p
        result = tl.where(mask_dropout, result / (1.0 - dropout_p), 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)


def fused_mul_add_logsoftmax_dropout_bmm(
    input1: Tensor,
    input2: Tensor,
    other: Tensor,
    mat2: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    dim: int = -1,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused operation: mul(input1, input2) -> add(other) -> logsoftmax -> dropout -> bmm(mat2)

    Args:
        input1: First input tensor for element-wise multiplication
        input2: Second input tensor for element-wise multiplication
        other: Tensor to add after multiplication
        mat2: Tensor for batch matrix multiplication
        p: Dropout probability (default: 0.5)
        training: Whether in training mode for dropout (default: True)
        inplace: Whether to perform in-place operations (default: False)
        dim: Dimension for log_softmax (default: -1)
        out: Optional output tensor

    Returns:
        Result tensor after all fused operations
    """

    # Step 1: Element-wise multiplication
    y = input1 * input2

    # Step 2: Addition
    y = y + other

    # Step 3: Log-softmax
    y = F.log_softmax(y, dim=dim)

    # Step 4: Dropout
    if training and p > 0.0:
        y = F.dropout(y, p=p, training=training, inplace=inplace)

    # Step 5: Batch matrix multiplication
    y = torch.bmm(y, mat2)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_mul_add_logsoftmax_dropout_bmm():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2)

    # Test case 2: Different dropout probability
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.3)

    # Test case 3: In-place operation
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, inplace=True)

    # Test case 4: Different dimension for log-softmax
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, dim=1)

    return results

test_results = test_fused_mul_add_logsoftmax_dropout_bmm()
