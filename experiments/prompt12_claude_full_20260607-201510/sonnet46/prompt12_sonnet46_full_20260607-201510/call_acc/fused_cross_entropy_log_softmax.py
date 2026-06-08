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


def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Computes cross entropy loss with log softmax applied along `dim`.
    Combines log_softmax and cross_entropy in a numerically stable way.
    """
    ndim = input.dim()
    # Normalize dim to positive
    if dim < 0:
        dim = dim + ndim

    # F.cross_entropy expects the class dimension at index 1 for N-D inputs,
    # or at index 0 for 1-D inputs. Handle the general case by moving `dim` to 1.
    if ndim == 1:
        # 1-D input: dim must be 0; F.cross_entropy handles scalar target
        # F.cross_entropy for 1-D input treats it as (C,) with a scalar target
        log_probs = F.log_softmax(input, dim=dim)
        # Use nll_loss for 1-D
        return F.nll_loss(
            log_probs.unsqueeze(0),
            target.unsqueeze(0) if target.dim() == 0 else target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        ).squeeze() if target.dim() == 0 else F.nll_loss(
            log_probs.unsqueeze(0),
            target.unsqueeze(0),
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    if dim == 1:
        # Standard case, use F.cross_entropy directly
        return F.cross_entropy(
            input,
            target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    else:
        # Move the class dimension to position 1
        # input shape: (..., C, ...) -> move dim to 1
        # Build permutation: [0, dim, 1, 2, ..., dim-1, dim+1, ..., ndim-1]
        perm = list(range(ndim))
        perm.pop(dim)
        perm.insert(1, dim)
        input_permuted = input.permute(perm)
        # input_permuted is now (N, C, d2, d3, ...) with class at dim 1
        return F.cross_entropy(
            input_permuted,
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
