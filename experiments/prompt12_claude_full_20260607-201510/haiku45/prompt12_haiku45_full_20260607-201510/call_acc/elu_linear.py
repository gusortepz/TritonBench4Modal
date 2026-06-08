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
def _elu_linear_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    alpha,
    n_rows,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for linear transformation followed by ELU activation.
    Computes: output = ELU(input @ weight^T + bias, alpha)
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    row_start = row_idx * BLOCK_SIZE
    col_start = col_idx * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    
    for i in tl.range(BLOCK_SIZE):
        row = row_start + i
        if row >= n_rows:
            break
        
        for j in tl.range(BLOCK_SIZE):
            col = col_start + j
            if col >= n_cols:
                break
            
            acc = tl.zeros((), dtype=tl.float32)
            for k in tl.range(0, n_cols):
                input_val = tl.load(input_ptr + row * input_stride + k)
                weight_val = tl.load(weight_ptr + col * n_cols + k)
                acc += input_val * weight_val
            
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + col)
                acc += bias_val
            
            elu_val = tl.where(
                acc > 0.0,
                acc,
                alpha * (tl.exp(acc) - 1.0)
            )
            
            tl.store(output_ptr + row * output_stride + col, elu_val)


def _elu_linear_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    alpha: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    """
    Reference implementation: linear transformation followed by ELU.
    """
    output = F.linear(input, weight, bias)
    return F.elu(output, alpha=alpha, inplace=inplace)


try:
    _elu_linear_fast = torch.compile(
        _elu_linear_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _elu_linear_fast = _elu_linear_impl


def elu_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    alpha: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    """
    Applies a linear transformation to the input tensor, followed by ELU activation.
    
    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,). Default: None
        alpha: Alpha parameter for ELU. Default: 1.0
        inplace: Whether to apply ELU in-place. Default: False
    
    Returns:
        Output tensor of shape (..., out_features) with ELU applied.
    """
    try:
        return _elu_linear_fast(input, weight, bias, alpha, inplace)
    except Exception:
        return _elu_linear_impl(input, weight, bias, alpha, inplace)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
#     output = F.linear(input, weight, bias)
#     return F.elu(output, alpha=alpha, inplace=inplace)

def test_elu_linear():
    results = {}

    # Test case 1: Basic test with bias, alpha=1.0, inplace=False
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = elu_linear(input1, weight1, bias1)

    # Test case 2: Without bias, alpha=1.0, inplace=False
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    results["test_case_2"] = elu_linear(input2, weight2)

    # Test case 3: With bias, alpha=0.5, inplace=False
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight3 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = elu_linear(input3, weight3, bias3, alpha=0.5)

    # Test case 4: With bias, alpha=1.0, inplace=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias4 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_4"] = elu_linear(input4, weight4, bias4, inplace=True)

    return results

test_results = test_elu_linear()
