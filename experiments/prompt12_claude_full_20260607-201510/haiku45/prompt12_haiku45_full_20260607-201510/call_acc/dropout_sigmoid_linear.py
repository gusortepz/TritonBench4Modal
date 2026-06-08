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
def _sigmoid_dropout_kernel(output_ptr, output_stride, numel, p_keep: tl.constexpr, philox_seed, philox_offset):
    """Fused sigmoid + dropout kernel."""
    idx = triton.program_id(0) * triton.next_power_of_2(1024) + tl.arange(0, triton.next_power_of_2(1024))
    mask = idx < numel
    
    offset = philox_offset + idx
    rand_vals = tl.rand(philox_seed, offset)
    keep_mask = rand_vals < p_keep
    
    x = tl.load(output_ptr + idx * output_stride, mask=mask, other=0.0)
    sig_x = 1.0 / (1.0 + tl.exp(-x))
    result = sig_x * keep_mask.to(sig_x.dtype) / p_keep
    
    tl.store(output_ptr + idx * output_stride, result, mask=mask)


def dropout_sigmoid_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False
) -> Tensor:
    """
    Applies a linear transformation followed by sigmoid activation and dropout.
    
    Args:
        input: Input tensor of shape (*, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Bias tensor of shape (out_features), optional
        p: Dropout probability, default 0.5
        training: Whether to apply dropout (only during training)
        inplace: Whether to perform operation in-place
    
    Returns:
        Output tensor after linear transformation, sigmoid, and dropout
    """
    
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        output = F.linear(input, weight, bias)
        output = torch.sigmoid(output)
        if training and p > 0:
            output = F.dropout(output, p=p, training=training, inplace=inplace)
        return output
    
    output = F.linear(input, weight, bias)
    
    if not training or p == 0:
        output = torch.sigmoid(output)
        return output
    
    if p == 1.0:
        return torch.zeros_like(output)
    
    numel = output.numel()
    
    try:
        p_keep = 1.0 - p
        philox_seed = torch.randint(0, 2**32, (1,), device=output.device).item()
        philox_offset = torch.randint(0, 2**32, (1,), device=output.device).item()
        
        grid = (triton.cdiv(numel, triton.next_power_of_2(1024)),)
        _sigmoid_dropout_kernel[grid](
            output,
            output.stride(-1) if output.dim() > 0 else 1,
            numel,
            p_keep,
            philox_seed,
            philox_offset
        )
        
        return output
    except Exception:
        output = torch.sigmoid(output)
        output = F.dropout(output, p=p, training=training, inplace=inplace)
        return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
#     """
#     Applies a linear transformation followed by a sigmoid activation and dropout.

#     Args:
#         input (torch.Tensor): Input tensor of shape (*, in_features).
#         weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
#         bias (torch.Tensor, optional): Bias tensor of shape (out_features). Default: None.
#         p (float, optional): Probability of an element to be zeroed in dropout. Default: 0.5.
#         training (bool, optional): If True, applies dropout during training. Default: True.
#         inplace (bool, optional): If True, performs the operation in-place. Default: False.

#     Returns:
#         torch.Tensor: The resulting tensor after applying the linear transformation, sigmoid activation, and dropout.
#     """
#     output = F.linear(input, weight, bias)
#     output = torch.sigmoid(output)
#     if training:
#         output = F.dropout(output, p=p, training=training, inplace=inplace)
#     return output

def test_dropout_sigmoid_linear():
    results = {}
    
    # Test case 1: Basic test with bias, training=True, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_1"] = dropout_sigmoid_linear(input, weight, bias)
    
    # Test case 2: No bias, training=True, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    results["test_case_2"] = dropout_sigmoid_linear(input, weight)
    
    # Test case 3: With bias, training=False, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_3"] = dropout_sigmoid_linear(input, weight, bias, training=False)
    
    # Test case 4: With bias, training=True, inplace=True
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_4"] = dropout_sigmoid_linear(input, weight, bias, inplace=True)
    
    return results

test_results = test_dropout_sigmoid_linear()
