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


def Adam(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False,
    foreach=None,
    maximize=False,
    capturable=False,
    differentiable=False,
    fused=None,
) -> torch.optim.Optimizer:
    """
    Creates and returns a torch.optim.Adam optimizer with the specified parameters.
    Implements the Adam optimization algorithm with optional AMSGrad variant.
    """
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=foreach,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
    )
    return optimizer

##################################################################################################################################################



import torch

# def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
#     return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def test_Adam():
    results = {}

    # Test Case 1: Default parameters
    params1 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer1 = Adam(params1)
    results["test_case_1"] = optimizer1.defaults

    # Test Case 2: Custom learning rate
    params2 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer2 = Adam(params2, lr=0.01)
    results["test_case_2"] = optimizer2.defaults

    # Test Case 3: Custom betas
    params3 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer3 = Adam(params3, betas=(0.85, 0.95))
    results["test_case_3"] = optimizer3.defaults

    # Test Case 4: Custom weight decay
    params4 = [torch.randn(2, 2, device='cuda', requires_grad=True)]
    optimizer4 = Adam(params4, weight_decay=0.01)
    results["test_case_4"] = optimizer4.defaults

    return results

test_results = test_Adam()
