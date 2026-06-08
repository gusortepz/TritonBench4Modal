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


def _sigmoid_adaptive_avg_pool2d_impl(
    input: Tensor,
    output_size: Union[int, Tuple[int, int]],
) -> Tensor:
    # Normalize output_size to a tuple
    if isinstance(output_size, int):
        os = (output_size, output_size)
    else:
        os = tuple(output_size)
    pooled = F.adaptive_avg_pool2d(input, os)
    return torch.sigmoid(pooled)


try:
    _sigmoid_adaptive_avg_pool2d_fast = torch.compile(
        _sigmoid_adaptive_avg_pool2d_impl,
        mode="max-autotune",
        fullgraph=False,
    )
except Exception:
    _sigmoid_adaptive_avg_pool2d_fast = _sigmoid_adaptive_avg_pool2d_impl


def sigmoid_adaptive_avg_pool2d(
    input: Tensor,
    output_size: Union[int, Tuple[int, int]],
) -> Tensor:
    try:
        return _sigmoid_adaptive_avg_pool2d_fast(input, output_size)
    except Exception:
        return _sigmoid_adaptive_avg_pool2d_impl(input, output_size)

##################################################################################################################################################



def test_sigmoid_adaptive_avg_pool2d():
    # Initialize a dictionary to store the results of each test case
    results = {}

    # Test case 1: Basic test with a 4D tensor and output size as an integer
    input_tensor1 = torch.randn(1, 3, 8, 8, device='cuda')  # Batch size 1, 3 channels, 8x8 size
    output_size1 = 4
    result1 = sigmoid_adaptive_avg_pool2d(input_tensor1, output_size1)
    results["test_case_1"] = result1

    # Test case 2: Test with a 4D tensor and output size as a tuple
    input_tensor2 = torch.randn(2, 3, 10, 10, device='cuda')  # Batch size 2, 3 channels, 10x10 size
    output_size2 = (5, 5)
    result2 = sigmoid_adaptive_avg_pool2d(input_tensor2, output_size2)
    results["test_case_2"] = result2

    # Test case 3: Test with a larger batch size
    input_tensor3 = torch.randn(4, 3, 16, 16, device='cuda')  # Batch size 4, 3 channels, 16x16 size
    output_size3 = (8, 8)
    result3 = sigmoid_adaptive_avg_pool2d(input_tensor3, output_size3)
    results["test_case_3"] = result3

    # Test case 4: Test with a single channel
    input_tensor4 = torch.randn(1, 1, 12, 12, device='cuda')  # Batch size 1, 1 channel, 12x12 size
    output_size4 = (6, 6)
    result4 = sigmoid_adaptive_avg_pool2d(input_tensor4, output_size4)
    results["test_case_4"] = result4

    return results

test_results = test_sigmoid_adaptive_avg_pool2d()
