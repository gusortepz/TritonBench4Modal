import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

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


def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Applies 2D convolution followed by batch normalization, ReLU activation, and dropout.
    
    Args:
        input: Input tensor of shape (N, C_in, H, W).
        weight: Convolution filters of shape (C_out, C_in / groups, kH, kW).
        bias: Optional bias tensor of shape (C_out). Default: None.
        stride: Stride of the convolution. Default: 1.
        padding: Implicit padding on both sides of the input. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        groups: Number of blocked connections from input channels to output channels. Default: 1.
        p: Probability of an element to be zeroed in dropout. Default: 0.5.
        training: If True, applies dropout during training. Default: True.
        inplace: If True, performs the operation in-place. Default: False.
    
    Returns:
        Output tensor.
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
    # Infer num_features from the output channels (weight.shape[0])
    num_features = weight.shape[0]
    # Use running_mean and running_var set to None for a simple BN operation
    bn_out = F.batch_norm(
        conv_out,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=training,
        momentum=0.1,
        eps=1e-5,
    )
    
    # Step 3: Apply ReLU activation
    relu_out = F.relu(bn_out, inplace=inplace)
    
    # Step 4: Apply dropout
    dropout_out = F.dropout(relu_out, p=p, training=training, inplace=False)
    
    return dropout_out

##################################################################################################################################################



def test_dropout_relu_batch_norm_conv2d():
    # Initialize test results dictionary
    test_results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(1, 3, 8, 8, device='cuda')
    weight_tensor = torch.randn(6, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(6, device='cuda')
    test_results["test_case_1"] = dropout_relu_batch_norm_conv2d(input_tensor, weight_tensor, bias_tensor)

    # Test case 2: Test with stride and padding
    test_results["test_case_2"] = dropout_relu_batch_norm_conv2d(input_tensor, weight_tensor, bias_tensor, stride=2, padding=1)

    # Test case 3: Test with different dropout probability
    test_results["test_case_3"] = dropout_relu_batch_norm_conv2d(input_tensor, weight_tensor, bias_tensor, p=0.3)

    # Test case 4: Test with groups
    weight_tensor_groups = torch.randn(6, 1, 3, 3, device='cuda')  # Adjust weight shape for groups
    input_tensor_groups = torch.randn(1, 6, 8, 8, device='cuda')   # Adjust input shape for groups
    test_results["test_case_4"] = dropout_relu_batch_norm_conv2d(input_tensor_groups, weight_tensor_groups, bias_tensor, groups=6)

    return test_results

# Execute the test function
test_results = test_dropout_relu_batch_norm_conv2d()
