import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
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


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor:
    """
    Applies Batch Normalization for each channel across a batch of data.
    
    Args:
        input: input tensor (N, C, ...) or (N, C)
        running_mean: running mean buffer (C,) for evaluation mode
        running_var: running variance buffer (C,) for evaluation mode
        weight: optional learnable scale parameter (C,)
        bias: optional learnable shift parameter (C,)
        training: whether in training mode
        momentum: momentum for running mean/var update
        eps: small value for numerical stability
    
    Returns:
        output tensor of same shape as input
    """
    if input.dim() < 2:
        raise ValueError("input must have at least 2 dimensions")
    
    num_channels = input.shape[1]
    
    # Infer normalized shape: last dimension for 2D, or all dims after batch+channel
    if input.dim() == 2:
        normalized_shape = (num_channels,)
    else:
        normalized_shape = tuple(input.shape[1:])
    
    if training:
        # Compute batch statistics
        # Reshape to (N, C, -1) for batch norm computation
        original_shape = input.shape
        x = input.view(input.shape[0], num_channels, -1)
        
        # Compute mean and var over (N, spatial_dims), keeping C
        mean = x.mean(dim=(0, 2), keepdim=False)  # (C,)
        var = x.var(dim=(0, 2), keepdim=False, unbiased=False)  # (C,)
        
        # Update running statistics if provided
        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + mean.detach() * momentum)
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + var.detach() * momentum)
        
        # Normalize using batch statistics
        x_norm = (x - mean.view(1, -1, 1)) / (torch.sqrt(var.view(1, -1, 1)) + eps)
        output = x_norm.view(original_shape)
    else:
        # Use running statistics
        if running_mean is None or running_var is None:
            raise ValueError(
                "running_mean and running_var must be provided in evaluation mode"
            )
        
        # Reshape and normalize
        original_shape = input.shape
        x = input.view(input.shape[0], num_channels, -1)
        mean = running_mean.view(1, -1, 1)
        var = running_var.view(1, -1, 1)
        
        x_norm = (x - mean) / (torch.sqrt(var) + eps)
        output = x_norm.view(original_shape)
    
    # Apply affine transformation
    if weight is not None or bias is not None:
        # Reshape weight/bias for broadcasting
        affine_shape = [1 if i != 1 else num_channels for i in range(output.dim())]
        
        if weight is not None:
            w = weight.view(affine_shape)
            output = output * w
        if bias is not None:
            b = bias.view(affine_shape)
            output = output + b
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
#     """
#     Applies Batch Normalization to each channel across a batch of data.
    
#     Parameters:
#         input (Tensor): Input tensor with shape (N, C, H, W) for 4D input (e.g., images).
#         running_mean (Tensor): Running mean for each channel, used in evaluation mode.
#         running_var (Tensor): Running variance for each channel, used in evaluation mode.
#         weight (Tensor, optional): Learnable scaling parameter for each channel.
#         bias (Tensor, optional): Learnable bias for each channel.
#         training (bool): Whether to use the statistics from the current batch or the running statistics.
#         momentum (float): The value used to update running_mean and running_var.
#         eps (float): A small value added to the denominator for numerical stability.

#     Returns:
#         Tensor: The normalized output.
#     """
#     return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def test_batch_norm():
    results = {}

    # Test case 1: Basic test with training=False
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_1"] = batch_norm(input, running_mean, running_var)

    # Test case 2: Test with training=True
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_2"] = batch_norm(input, running_mean, running_var, training=True)

    # Test case 3: Test with weight and bias
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    weight = torch.randn(3, device='cuda')
    bias = torch.randn(3, device='cuda')
    results["test_case_3"] = batch_norm(input, running_mean, running_var, weight, bias)

    # Test case 4: Test with different momentum and eps
    input = torch.randn(2, 3, 4, 4, device='cuda')
    running_mean = torch.zeros(3, device='cuda')
    running_var = torch.ones(3, device='cuda')
    results["test_case_4"] = batch_norm(input, running_mean, running_var, momentum=0.2, eps=1e-03)

    return results

test_results = test_batch_norm()
