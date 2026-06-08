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


def _fused_transformer_block_impl(
    input: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    residual: Tensor,
    dropout_p: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    # Step 1: First matmul: (..., N, D_in) @ (D_in, D_k) -> (..., N, D_k)
    z1 = torch.matmul(input, weight1)

    # Step 2: Softmax over last dimension
    z2 = F.softmax(z1, dim=-1)

    # Step 3: Dropout
    z3 = F.dropout(z2, p=dropout_p, training=True)

    # Step 4: Second matmul: (..., N, D_k) @ (D_k, D_out) -> (..., N, D_out)
    z4 = torch.matmul(z3, weight2)

    # Step 5: Add residual (must be broadcastable to z4's shape)
    z5 = z4 + residual

    # Step 6: Layer normalization over last dimension
    normalized_shape = (z5.shape[-1],)
    result = F.layer_norm(z5, normalized_shape, weight=None, bias=None, eps=eps)

    return result


def fused_transformer_block(
    input: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    residual: Tensor,
    dropout_p: float = 0.1,
    eps: float = 1e-5,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Performs a sequence of operations commonly used in transformer models:
    matmul -> softmax -> dropout -> matmul -> add residual -> layer norm.

    Args:
        input: Input tensor of shape (*, N, D_in)
        weight1: Weight matrix of shape (D_in, D_k)
        weight2: Weight matrix of shape (D_k, D_out)
        residual: Residual tensor broadcastable to output of second matmul
        dropout_p: Dropout probability (default: 0.1)
        eps: Layer norm epsilon (default: 1e-5)
        out: Optional output tensor

    Returns:
        Result tensor after all operations
    """
    y = _fused_transformer_block_impl(
        input=input,
        weight1=weight1,
        weight2=weight2,
        residual=residual,
        dropout_p=dropout_p,
        eps=eps,
    )

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_transformer_block():
    results = {}

    # Test case 1: Basic functionality test
    input1 = torch.randn(2, 3, 4, device='cuda')
    weight1_1 = torch.randn(4, 5, device='cuda')
    weight2_1 = torch.randn(5, 4, device='cuda')
    residual1 = torch.randn(2, 3, 4, device='cuda')
    results["test_case_1"] = fused_transformer_block(input1, weight1_1, weight2_1, residual1)

    # Test case 2: Different input size
    input2 = torch.randn(1, 5, 6, device='cuda')
    weight1_2 = torch.randn(6, 7, device='cuda')
    weight2_2 = torch.randn(7, 6, device='cuda')
    residual2 = torch.randn(1, 5, 6, device='cuda')
    results["test_case_2"] = fused_transformer_block(input2, weight1_2, weight2_2, residual2)

    # Test case 3: Test with dropout probability set to 0
    input3 = torch.randn(3, 2, 4, device='cuda')
    weight1_3 = torch.randn(4, 5, device='cuda')
    weight2_3 = torch.randn(5, 4, device='cuda')
    residual3 = torch.randn(3, 2, 4, device='cuda')
    results["test_case_3"] = fused_transformer_block(input3, weight1_3, weight2_3, residual3, dropout_p=0.0)

    # Test case 4: Test with a different epsilon value
    input4 = torch.randn(4, 3, 5, device='cuda')
    weight1_4 = torch.randn(5, 6, device='cuda')
    weight2_4 = torch.randn(6, 5, device='cuda')
    residual4 = torch.randn(4, 3, 5, device='cuda')
    results["test_case_4"] = fused_transformer_block(input4, weight1_4, weight2_4, residual4, eps=1e-3)

    return results

test_results = test_fused_transformer_block()
