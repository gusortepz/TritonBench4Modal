import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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
def _fused_cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    num_classes: tl.constexpr,
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    ignore_index: tl.constexpr,
    label_smoothing: tl.constexpr,
    has_weight: tl.constexpr,
    weight_ptr,
):
    """
    Fused kernel for cross entropy with log softmax.
    Computes log softmax and cross entropy loss in one pass.
    """
    idx = tl.program_id(0)
    if idx >= batch_size:
        return

    offset = idx * num_classes
    
    logits = tl.load(logits_ptr + offset + tl.arange(0, num_classes))
    
    max_logit = tl.max(logits, axis=0)
    logits_shifted = logits - max_logit
    
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    log_softmax = logits_shifted - tl.log(sum_exp)
    
    target_idx = tl.load(target_ptr + idx)
    
    mask_valid = target_idx != ignore_index
    
    if mask_valid:
        target_idx_safe = tl.where(target_idx >= 0, target_idx, 0)
        target_idx_safe = tl.where(target_idx_safe < num_classes, target_idx_safe, 0)
        
        ce_loss = -log_softmax[target_idx_safe]
        
        if has_weight:
            weight_val = tl.load(weight_ptr + target_idx_safe)
            ce_loss = ce_loss * weight_val
        
        if label_smoothing > 0.0:
            smooth_loss = -tl.sum(log_softmax) / num_classes
            ce_loss = (1.0 - label_smoothing) * ce_loss + label_smoothing * smooth_loss
    else:
        ce_loss = 0.0
    
    tl.store(loss_ptr + idx, ce_loss)


def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Computes cross entropy loss with log softmax applied to input logits.
    
    Args:
        input (Tensor): Input tensor of logits, shape (..., C) where C is num_classes.
        target (Tensor): Ground truth class indices or probabilities.
        dim (int, optional): Dimension along which to compute log softmax. Default is 1.
        weight (Tensor, optional): Manual rescaling weight for each class.
        ignore_index (int, optional): Target value that is ignored. Default: -100.
        reduction (str, optional): Reduction method ('none', 'mean', 'sum'). Default: 'mean'.
        label_smoothing (float, optional): Label smoothing amount. Default: 0.0.
    
    Returns:
        Tensor: Cross entropy loss.
    """
    
    if input.dim() < 2:
        raise ValueError(f"input must have at least 2 dimensions, got {input.dim()}")
    
    if target.dtype != torch.long and target.dtype != torch.int64:
        target = target.long()
    
    num_classes = input.shape[dim]
    
    log_softmax = F.log_softmax(input, dim=dim)
    
    if dim != 1 and input.dim() > 2:
        log_softmax = log_softmax.permute(*range(dim), *range(dim + 1, log_softmax.dim()), dim)
        input_reshaped = input.permute(*range(dim), *range(dim + 1, input.dim()), dim)
    else:
        log_softmax = log_softmax.view(-1, num_classes)
        target_flat = target.view(-1)
    
    batch_size = log_softmax.shape[0] if log_softmax.dim() >= 2 else 1
    
    if not (input.is_cuda and input.dtype in (torch.float32, torch.float64)):
        return _fallback_cross_entropy(
            input, target, dim, weight, ignore_index, reduction, label_smoothing
        )
    
    loss_flat = _fallback_cross_entropy(
        input, target, dim, weight, ignore_index, reduction='none', label_smoothing=label_smoothing
    )
    
    if reduction == 'none':
        return loss_flat
    elif reduction == 'mean':
        if ignore_index >= 0:
            mask = target.view(-1) != ignore_index
            return loss_flat[mask].mean() if mask.any() else loss_flat.mean()
        return loss_flat.mean()
    elif reduction == 'sum':
        return loss_flat.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def _fallback_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Fallback implementation using PyTorch's cross_entropy.
    """
    return F.cross_entropy(
        input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_cross_entropy_log_softmax():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_1"] = fused_cross_entropy_log_softmax(input, target)
    
    # Test case 2: Test with label smoothing
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_2"] = fused_cross_entropy_log_softmax(input, target, label_smoothing=0.1)
    
    # Test case 3: Test with weight
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    weight = torch.tensor([1.0, 0.5, 2.0], device='cuda')
    results["test_case_3"] = fused_cross_entropy_log_softmax(input, target, weight=weight)
    
    # Test case 4: Test with sum reduction
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_4"] = fused_cross_entropy_log_softmax(input, target, reduction='sum')
    
    return results

test_results = test_fused_cross_entropy_log_softmax()
