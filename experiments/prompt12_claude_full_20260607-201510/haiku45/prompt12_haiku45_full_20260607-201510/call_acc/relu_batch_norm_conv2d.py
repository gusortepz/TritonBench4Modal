import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union

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


def relu_batch_norm_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple, str] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    bn_weight: Optional[Tensor] = None,
    bn_bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False,
) -> Tensor:
    """
    Applies a 2D convolution over the input tensor, followed by batch normalization
    and then applies the ReLU activation function element-wise to the normalized result.
    """
    
    # Step 1: Apply 2D convolution
    conv_out = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    
    # Step 2: Apply batch normalization
    # Determine the number of channels (features) from the conv output
    num_features = conv_out.shape[1]
    
    # Prepare batch norm parameters with safety checks
    bn_w = bn_weight if bn_weight is not None and bn_weight.shape[0] == num_features else None
    bn_b = bn_bias if bn_bias is not None and bn_bias.shape[0] == num_features else None
    
    # Apply batch normalization
    if training:
        # In training mode, compute statistics from the batch
        bn_out = F.batch_norm(
            conv_out,
            running_mean=running_mean,
            running_var=running_var,
            weight=bn_w,
            bias=bn_b,
            training=True,
            momentum=momentum,
            eps=eps,
        )
    else:
        # In eval mode, use running statistics
        bn_out = F.batch_norm(
            conv_out,
            running_mean=running_mean,
            running_var=running_var,
            weight=bn_w,
            bias=bn_b,
            training=False,
            momentum=momentum,
            eps=eps,
        )
    
    # Step 3: Apply ReLU activation
    if inplace:
        output = F.relu(bn_out, inplace=True)
    else:
        output = F.relu(bn_out, inplace=False)
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import nn

import torch
from torch import nn

# Define a simple test function
def test_relu_batch_norm_conv2d():
    # Define input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(4, 3, 32, 32)  # Example: 4 images, 3 channels, 32x32 resolution
    
    # Define convolution weight tensor (out_channels, in_channels/groups, kernel_height, kernel_width)
    weight_tensor = torch.randn(6, 3, 3, 3)  # Example: 6 filters, 3 input channels, 3x3 kernel
    
    # Define optional bias tensor (out_channels)
    bias_tensor = torch.randn(6)  # Example: bias for each of the 6 filters
    
    # Define batch normalization parameters
    running_mean = torch.zeros(6)
    running_var = torch.ones(6)
    bn_weight = torch.ones(6)
    bn_bias = torch.zeros(6)
    
    # Call the relu_batch_norm_conv2d function
    output_tensor = relu_batch_norm_conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        running_mean=running_mean,
        running_var=running_var,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
        inplace=False
    )

    # Print the shape of the output tensor
    print(f"Output tensor shape: {output_tensor.shape}")
    
    # Check if output tensor has the expected shape
    expected_shape = (4, 6, 32, 32)  # 4 images, 6 output channels, 32x32 resolution
    assert output_tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {output_tensor.shape}"

    return output_tensor

# Run the test
output = test_relu_batch_norm_conv2d()