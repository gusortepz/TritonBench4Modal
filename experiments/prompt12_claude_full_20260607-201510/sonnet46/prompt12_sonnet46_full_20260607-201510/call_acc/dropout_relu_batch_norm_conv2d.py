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


def _dropout_relu_bn_conv_impl(
    input: Tensor,
    weight: Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    p=0.5,
    training=True,
    inplace=False,
) -> Tensor:
    # Step 1: conv2d
    out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding,
                   dilation=dilation, groups=groups)

    # Step 2: batch normalization
    C_out = out.shape[1]
    bn_weight = torch.ones(C_out, device=out.device, dtype=out.dtype)
    bn_bias = torch.zeros(C_out, device=out.device, dtype=out.dtype)
    out = F.batch_norm(out, running_mean=None, running_var=None,
                       weight=bn_weight, bias=bn_bias,
                       training=True, eps=1e-5)

    # Step 3: ReLU
    out = F.relu(out, inplace=inplace)

    # Step 4: Dropout
    out = F.dropout(out, p=p, training=training, inplace=inplace)

    return out


try:
    _dropout_relu_bn_conv_fast = torch.compile(
        _dropout_relu_bn_conv_impl, mode="max-autotune", fullgraph=False
    )
except Exception:
    _dropout_relu_bn_conv_fast = _dropout_relu_bn_conv_impl


def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    p=0.5,
    training=True,
    inplace=False,
) -> torch.Tensor:
    try:
        return _dropout_relu_bn_conv_fast(
            input, weight, bias=bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups, p=p, training=training,
            inplace=inplace
        )
    except Exception:
        return _dropout_relu_bn_conv_impl(
            input, weight, bias=bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups, p=p, training=training,
            inplace=inplace
        )

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
