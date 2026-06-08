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
def _fused_mul_add_kernel(
    input1_ptr,
    input2_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x1 = tl.load(input1_ptr + offsets, mask=mask)
    x2 = tl.load(input2_ptr + offsets, mask=mask)
    x3 = tl.load(other_ptr + offsets, mask=mask)

    result = x1 * x2 + x3

    tl.store(output_ptr + offsets, result, mask=mask)


def _triton_mul_add(input1: Tensor, input2: Tensor, other: Tensor) -> Tensor:
    """Fused elementwise multiply-add using Triton."""
    output = torch.empty_like(input1)
    n_elements = input1.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _fused_mul_add_kernel[grid](
        input1,
        input2,
        other,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def _pytorch_mul_add(input1: Tensor, input2: Tensor, other: Tensor) -> Tensor:
    return input1 * input2 + other


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
    # Step 1: element-wise mul + add (fused with Triton when possible)
    use_triton = (
        input1.is_cuda
        and input2.is_cuda
        and other.is_cuda
        and input1.is_contiguous()
        and input2.is_contiguous()
        and other.is_contiguous()
        and input1.shape == input2.shape
        and input1.shape == other.shape
        and input1.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and not input1.is_complex()
    )

    if use_triton:
        try:
            mul_add_result = _triton_mul_add(
                input1.contiguous(),
                input2.contiguous(),
                other.contiguous(),
            )
        except Exception:
            mul_add_result = _pytorch_mul_add(input1, input2, other)
    else:
        mul_add_result = _pytorch_mul_add(input1, input2, other)

    # Step 2: log-softmax
    log_softmax_result = F.log_softmax(mul_add_result, dim=dim)

    # Step 3: dropout
    dropout_result = F.dropout(log_softmax_result, p=p, training=training, inplace=inplace)

    # Step 4: batch matrix multiplication
    # dropout_result needs to be 3D for bmm
    # Handle shape compatibility
    if dropout_result.dim() == 2:
        dropout_result = dropout_result.unsqueeze(0)
        y = torch.bmm(dropout_result, mat2)
        y = y.squeeze(0)
    elif dropout_result.dim() == 3:
        y = torch.bmm(dropout_result, mat2)
    else:
        # Flatten to 3D for bmm if needed
        orig_shape = dropout_result.shape
        # Treat all dims except last two as batch
        batch_dims = orig_shape[:-2]
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        m, k = orig_shape[-2], orig_shape[-1]
        dropout_3d = dropout_result.reshape(batch_size, m, k)
        y = torch.bmm(dropout_3d, mat2)
        # Restore batch dimensions
        out_shape = batch_dims + (m, mat2.shape[-1])
        y = y.reshape(out_shape)

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
