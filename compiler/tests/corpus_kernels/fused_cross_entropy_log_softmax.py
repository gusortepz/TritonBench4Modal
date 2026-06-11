import triton
import triton.language as tl


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

