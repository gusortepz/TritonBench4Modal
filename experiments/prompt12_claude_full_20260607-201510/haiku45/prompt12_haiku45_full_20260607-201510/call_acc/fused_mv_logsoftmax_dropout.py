import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
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


def fused_mv_logsoftmax_dropout(
    input: Tensor,
    vec: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    dim: int = 0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused operation combining matrix-vector multiplication, log-softmax activation, and dropout.
    
    Args:
        input: Input matrix of shape (m, n) or (n, m)
        vec: Vector of shape (n,) or (m,)
        p: Dropout probability (default: 0.5)
        training: Whether in training mode (default: True)
        inplace: Whether to apply operations in-place (default: False)
        dim: Dimension along which to apply log-softmax (default: 0)
        out: Optional output tensor
    
    Returns:
        Output tensor with log-softmax and dropout applied
    """
    # Step 1: Matrix-vector multiplication
    mv_result = torch.mv(input, vec)
    
    # Step 2: Log-softmax activation
    logsoftmax_result = F.log_softmax(mv_result, dim=dim)
    
    # Step 3: Dropout
    y = F.dropout(logsoftmax_result, p=p, training=training, inplace=inplace)
    
    # Handle optional output tensor
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_mv_logsoftmax_dropout():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.randn(3, 4, device='cuda')
    vec1 = torch.randn(4, device='cuda')
    results["test_case_1"] = fused_mv_logsoftmax_dropout(input1, vec1)

    # Test case 2: Dropout with p=0.2
    input2 = torch.randn(3, 4, device='cuda')
    vec2 = torch.randn(4, device='cuda')
    results["test_case_2"] = fused_mv_logsoftmax_dropout(input2, vec2, p=0.2)

    # Test case 3: Dropout in evaluation mode (training=False)
    input3 = torch.randn(3, 4, device='cuda')
    vec3 = torch.randn(4, device='cuda')
    results["test_case_3"] = fused_mv_logsoftmax_dropout(input3, vec3, training=False)

    # Test case 4: Inplace operation
    input4 = torch.randn(3, 4, device='cuda')
    vec4 = torch.randn(4, device='cuda')
    results["test_case_4"] = fused_mv_logsoftmax_dropout(input4, vec4, inplace=True)

    return results

test_results = test_fused_mv_logsoftmax_dropout()
