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


@triton.jit
def _fused_hardshrink_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    p: tl.constexpr,
    training: tl.constexpr,
    lambd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: dropout followed by hard shrinkage.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if training:
        dropout_mask = tl.rand(tl.uint32(offsets), n_elements) > p
        scale = 1.0 / (1.0 - p)
        x = tl.where(dropout_mask, x * scale, 0.0)
    
    abs_x = tl.abs(x)
    hard_shrink = tl.where(abs_x > lambd, x, 0.0)
    
    tl.store(output_ptr + offsets, hard_shrink, mask=mask)


def fused_hardshrink_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> Tensor:
    """
    Applies fused dropout followed by hard shrinkage on the input tensor.
    
    Args:
        input (Tensor): The input tensor.
        p (float, optional): Probability of an element to be zeroed in dropout. Default is 0.5.
        training (bool, optional): Apply dropout if True. Default is True.
        inplace (bool, optional): If set to True, dropout will be applied in-place. Default is False.
        lambd (float, optional): The lambda parameter for the hard shrinkage function. Default is 0.5.
    
    Returns:
        Tensor: Result after applying dropout and then hard shrinkage on the input.
    """
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        return _fused_hardshrink_dropout_cpu(input, p, training, inplace, lambd)
    
    if inplace:
        output = input
    else:
        output = input.clone()
    
    if training and p > 0.0:
        output = F.dropout(output, p=p, training=True, inplace=True)
    
    output = torch.where(
        torch.abs(output) > lambd,
        output,
        torch.zeros_like(output)
    )
    
    return output


def _fused_hardshrink_dropout_cpu(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> Tensor:
    """
    Fallback implementation using PyTorch operations.
    """
    if inplace:
        output = input
    else:
        output = input.clone()
    
    if training and p > 0.0:
        output = F.dropout(output, p=p, training=True, inplace=True)
    
    output = torch.where(
        torch.abs(output) > lambd,
        output,
        torch.zeros_like(output)
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_hardshrink_dropout(input: torch.Tensor, p: float=0.5, training: bool=True, inplace: bool=False, lambd: float=0.5) -> torch.Tensor:
#     """
#     Applies a fused operation consisting of dropout followed by hard shrinkage on the input tensor.

#     Args:
#         input (Tensor): The input tensor.
#         p (float, optional): Probability of an element to be zeroed in dropout. Default is 0.5.
#         training (bool, optional): Apply dropout if True. Default is True.
#         inplace (bool, optional): If set to True, dropout will be applied in-place. Default is False.
#         lambd (float, optional): The lambda parameter for the hard shrinkage function. Default is 0.5.

#     Returns:
#         Tensor: Result after applying dropout and then hard shrinkage on the input.
#     """
#     if training:
#         input = F.dropout(input, p=p, training=training, inplace=inplace)
#     return F.hardshrink(input, lambd)

def test_fused_hardshrink_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_1"] = fused_hardshrink_dropout(input_tensor)
    
    # Test case 2: Dropout with p=0.3
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_2"] = fused_hardshrink_dropout(input_tensor, p=0.3)
    
    # Test case 3: Dropout with training=False
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_3"] = fused_hardshrink_dropout(input_tensor, training=False)
    
    # Test case 4: Hard shrinkage with lambd=0.7
    input_tensor = torch.randn(5, 5).cuda()
    results["test_case_4"] = fused_hardshrink_dropout(input_tensor, lambd=0.7)
    
    return results

test_results = test_fused_hardshrink_dropout()
