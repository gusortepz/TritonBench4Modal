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


def _fused_mv_logsoftmax_dropout_impl(
    input: Tensor,
    vec: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    dim: int = 0,
) -> Tensor:
    # Step 1: matrix-vector multiplication
    mv_result = torch.mv(input, vec)
    # Step 2: log-softmax along specified dimension
    ls_result = F.log_softmax(mv_result, dim=dim)
    # Step 3: dropout
    out = F.dropout(ls_result, p=p, training=training, inplace=inplace)
    return out


try:
    _fused_mv_logsoftmax_dropout_fast = torch.compile(
        _fused_mv_logsoftmax_dropout_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _fused_mv_logsoftmax_dropout_fast = _fused_mv_logsoftmax_dropout_impl


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
    try:
        y = _fused_mv_logsoftmax_dropout_fast(input, vec, p=p, training=training, inplace=inplace, dim=dim)
    except Exception:
        y = _fused_mv_logsoftmax_dropout_impl(input, vec, p=p, training=training, inplace=inplace, dim=dim)

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
