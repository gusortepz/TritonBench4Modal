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


def _softmax_layernorm_impl(
    logits: Tensor,
    normalized_shape,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    # Determine safe normalized_shape
    if isinstance(normalized_shape, int):
        shape = (normalized_shape,)
    else:
        shape = tuple(normalized_shape)
    # Validate weight/bias shapes
    w = weight if weight is not None and tuple(weight.shape) == shape else None
    b = bias if bias is not None and tuple(bias.shape) == shape else None
    normed = F.layer_norm(probs, shape, weight=w, bias=b, eps=eps)
    return normed


try:
    _softmax_layernorm_fast = torch.compile(
        _softmax_layernorm_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _softmax_layernorm_fast = _softmax_layernorm_impl


def fused_cross_entropy_softmax_layernorm(
    logits: Tensor,
    targets: Tensor,
    normalized_shape,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    eps: float = 1e-5,
    *,
    out: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    # Step 1: Compute cross-entropy loss
    # F.cross_entropy expects logits of shape (N, C) or (N, C, d1, d2, ...)
    # targets of shape (N,) or (N, d1, d2, ...) for class indices
    # or same shape as logits for probabilities
    loss = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    # Step 2 & 3: Softmax then LayerNorm
    # weight here is per-class rescaling for cross_entropy, not for layer_norm
    # For layer_norm we use no affine parameters unless explicitly provided
    # (The signature doesn't have separate layernorm weight/bias, so we use None)
    try:
        normed = _softmax_layernorm_fast(logits, normalized_shape, None, None, eps)
    except Exception:
        normed = _softmax_layernorm_impl(logits, normalized_shape, None, None, eps)

    # Handle out parameter
    if out is not None:
        out.copy_(normed)
        return loss, out

    return loss, normed

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_cross_entropy_softmax_layernorm():
    results = {}

    # Test case 1: Basic functionality with default parameters
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    targets = torch.tensor([2, 1], device='cuda')
    normalized_shape = 3
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape)
    results["test_case_1"] = (loss.item(), output.cpu().numpy())

    # Test case 2: With weight parameter
    weight = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=weight)
    results["test_case_2"] = (loss.item(), output.cpu().numpy())

    # Test case 3: With ignore_index parameter
    targets_ignore = torch.tensor([2, -100], device='cuda')
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets_ignore, normalized_shape, ignore_index=-100)
    results["test_case_3"] = (loss.item(), output.cpu().numpy())

    # Test case 4: With label_smoothing parameter
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, label_smoothing=0.1)
    results["test_case_4"] = (loss.item(), output.cpu().numpy())

    return results

test_results = test_fused_cross_entropy_softmax_layernorm()
