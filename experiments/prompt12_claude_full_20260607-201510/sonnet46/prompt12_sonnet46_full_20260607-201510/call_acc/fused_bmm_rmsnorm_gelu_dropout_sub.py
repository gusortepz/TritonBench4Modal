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


def _fused_impl(
    input1: Tensor,
    input2: Tensor,
    other: Tensor,
    normalized_shape,
    dropout_p: float,
    training: bool,
    approximate: str,
    eps: float,
) -> Tensor:
    # Step 1: Batch matrix multiplication
    y = torch.bmm(input1, input2)

    # Step 2: RMS Normalization
    # Compute normalized_shape safely
    if isinstance(normalized_shape, int):
        norm_shape = (normalized_shape,)
    else:
        norm_shape = tuple(normalized_shape)

    # RMS norm: y / rms(y) * weight (no weight/bias here)
    # rms = sqrt(mean(y^2) + eps)
    # We need to normalize over the last len(norm_shape) dimensions
    ndim = len(norm_shape)
    # Verify norm_shape matches last ndim dims of y
    y_norm_shape = tuple(y.shape[-ndim:])
    if y_norm_shape == norm_shape:
        # Compute RMS norm manually
        dims = tuple(range(-ndim, 0))
        rms = torch.sqrt(torch.mean(y * y, dim=dims, keepdim=True) + eps)
        y = y / rms
    else:
        # Fallback: normalize over last dim
        dims = (-1,)
        rms = torch.sqrt(torch.mean(y * y, dim=dims, keepdim=True) + eps)
        y = y / rms

    # Step 3: GELU activation
    y = F.gelu(y, approximate=approximate)

    # Step 4: Dropout
    if training and dropout_p > 0.0:
        y = F.dropout(y, p=dropout_p, training=True)

    # Step 5: Subtract other
    y = y - other

    return y


try:
    _fused_fast = torch.compile(_fused_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_fast = _fused_impl


def fused_bmm_rmsnorm_gelu_dropout_sub(
    input1: Tensor,
    input2: Tensor,
    other: Tensor,
    normalized_shape,
    dropout_p: float = 0.5,
    training: bool = True,
    approximate: str = 'none',
    eps: float = 1e-5,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    try:
        y = _fused_fast(
            input1,
            input2,
            other,
            normalized_shape,
            dropout_p,
            training,
            approximate,
            eps,
        )
    except Exception:
        y = _fused_impl(
            input1,
            input2,
            other,
            normalized_shape,
            dropout_p,
            training,
            approximate,
            eps,
        )

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_bmm_rmsnorm_gelu_dropout_sub():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    other = torch.randn(2, 3, 5, device='cuda')
    normalized_shape = 5
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)

    # Test case 2: Test with different dropout probability
    dropout_p = 0.3
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=dropout_p)

    # Test case 3: Test with training set to False
    training = False
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, training=training)

    # Test case 4: Test with approximate GELU
    approximate = 'tanh'
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, approximate=approximate)

    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout_sub()
