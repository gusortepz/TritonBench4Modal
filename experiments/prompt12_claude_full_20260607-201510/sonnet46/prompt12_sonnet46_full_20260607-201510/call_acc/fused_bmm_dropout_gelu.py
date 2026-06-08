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


def _fused_bmm_dropout_gelu_impl(
    input1: Tensor,
    input2: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    approximate: str = 'none',
) -> Tensor:
    # Step 1: batch matrix multiplication
    result = torch.bmm(input1, input2)
    # Step 2: dropout
    result = F.dropout(result, p=p, training=training, inplace=inplace)
    # Step 3: GELU activation
    result = F.gelu(result, approximate=approximate)
    return result


try:
    _fused_bmm_dropout_gelu_fast = torch.compile(
        _fused_bmm_dropout_gelu_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _fused_bmm_dropout_gelu_fast = _fused_bmm_dropout_gelu_impl


def fused_bmm_dropout_gelu(
    input1: Tensor,
    input2: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    approximate: str = 'none',
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    try:
        y = _fused_bmm_dropout_gelu_fast(
            input1, input2, p=p, training=training, inplace=inplace, approximate=approximate
        )
    except Exception:
        y = _fused_bmm_dropout_gelu_impl(
            input1, input2, p=p, training=training, inplace=inplace, approximate=approximate
        )

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
#     Z = torch.bmm(input1, input2)
#     D = torch.nn.functional.dropout(Z, p=p, training=training, inplace=inplace)
#     O = torch.nn.functional.gelu(D, approximate=approximate)
#     if out is not None:
#         out.copy_(O)
#         return out
#     return O

def test_fused_bmm_dropout_gelu():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_dropout_gelu(input1, input2)
    
    # Test case 2: Dropout with p=0.3 and training=False
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_bmm_dropout_gelu(input1, input2, p=0.3, training=False)
    
    # Test case 3: In-place dropout
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_bmm_dropout_gelu(input1, input2, inplace=True)
    
    # Test case 4: GELU with tanh approximation
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_bmm_dropout_gelu(input1, input2, approximate='tanh')
    
    return results

test_results = test_fused_bmm_dropout_gelu()
