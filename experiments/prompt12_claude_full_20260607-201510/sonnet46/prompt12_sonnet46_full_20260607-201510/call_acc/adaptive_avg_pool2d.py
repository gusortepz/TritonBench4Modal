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


def adaptive_avg_pool2d(input: Tensor, output_size) -> Tensor:
    """
    Apply a 2D adaptive average pooling over an input signal.
    
    Args:
        input: Input tensor of shape (N, C, H_in, W_in) or (C, H_in, W_in)
        output_size: the target output size (single integer or double-integer tuple)
    
    Returns:
        Output tensor of shape (N, C, S_0, S_1) or (C, S_0, S_1)
    """
    return F.adaptive_avg_pool2d(input, output_size)

##################################################################################################################################################



import torch
from adaptive_avg_pool2d import adaptive_avg_pool2d

def test_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: input shape (N, C, H_in, W_in), output_size as integer
    input1 = torch.randn(1, 64, 8, 9).cuda()
    output1 = adaptive_avg_pool2d(input1, 5)
    results["test_case_1"] = output1
    
    # Test case 2: input shape (N, C, H_in, W_in), output_size as tuple
    input2 = torch.randn(1, 64, 8, 9).cuda()
    output2 = adaptive_avg_pool2d(input2, (5, 7))
    results["test_case_2"] = output2
    
    # Test case 3: input shape (C, H_in, W_in), output_size as integer
    input3 = torch.randn(64, 8, 9).cuda()
    output3 = adaptive_avg_pool2d(input3, 5)
    results["test_case_3"] = output3
    
    # Test case 4: input shape (C, H_in, W_in), output_size as tuple
    input4 = torch.randn(64, 8, 9).cuda()
    output4 = adaptive_avg_pool2d(input4, (5, 7))
    results["test_case_4"] = output4
    
    return results

test_results = test_adaptive_avg_pool2d()
