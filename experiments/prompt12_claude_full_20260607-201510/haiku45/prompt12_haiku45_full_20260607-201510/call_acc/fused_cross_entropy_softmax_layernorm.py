import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple
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
    """
    Fused operation: cross-entropy loss + softmax + layer normalization.
    
    Returns:
        Tuple[Tensor, Tensor]: (loss, normalized_probs)
    """
    
    # Validate inputs
    if logits.dim() < 2:
        raise ValueError(f"logits must have at least 2 dimensions, got {logits.dim()}")
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Ensure targets is the correct type
    if targets.dtype in (torch.long, torch.int32, torch.int64):
        # targets are class indices
        targets_are_indices = True
    else:
        # targets are probabilities
        targets_are_indices = False
    
    # Compute softmax
    softmax_probs = F.softmax(logits, dim=1)
    
    # Compute cross-entropy loss
    if targets_are_indices:
        # Use standard cross-entropy for class indices
        loss = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        )
    else:
        # Use kl_div for probability targets
        log_softmax_logits = F.log_softmax(logits, dim=1)
        kl_loss = F.kl_div(
            log_softmax_logits,
            targets,
            reduction="none",
            log_target=False,
        )
        loss = kl_loss.sum(dim=1)
        
        # Apply ignore_index mask if needed (only for index-based targets)
        if ignore_index >= 0 and ignore_index < num_classes:
            mask = (targets.argmax(dim=1) != ignore_index).float()
            loss = loss * mask
    
    # Apply reduction to loss
    if reduction == "mean":
        if targets_are_indices:
            # Mask out ignored indices
            mask = (targets != ignore_index).float()
            valid_count = mask.sum()
            loss = (loss * mask).sum() / (valid_count + 1e-8)
        else:
            loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        # Keep as is
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    # Layer normalization on softmax_probs
    # Infer normalized_shape safely
    if isinstance(normalized_shape, int):
        norm_shape = (normalized_shape,)
    elif isinstance(normalized_shape, (list, tuple)):
        norm_shape = tuple(normalized_shape)
    elif isinstance(normalized_shape, torch.Size):
        norm_shape = tuple(normalized_shape)
    else:
        norm_shape = (softmax_probs.shape[-1],)
    
    # Validate weight shape if provided
    w = None
    b = None
    if weight is not None:
        if tuple(weight.shape) == norm_shape:
            w = weight
    
    normalized_probs = F.layer_norm(
        softmax_probs,
        norm_shape,
        weight=w,
        bias=b,
        eps=eps,
    )
    
    # Handle out parameter
    if out is not None:
        out.copy_(normalized_probs)
        return loss, out
    
    return loss, normalized_probs

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
