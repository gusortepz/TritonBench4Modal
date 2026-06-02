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

def _sigmoid_batch_norm_impl(input, running_mean, running_var, weight, bias, training, momentum, eps):
    out = F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    return torch.sigmoid(out)

try:
    _sigmoid_batch_norm_fast = torch.compile(_sigmoid_batch_norm_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _sigmoid_batch_norm_fast = _sigmoid_batch_norm_impl

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5, *, out=None):
    y = _sigmoid_batch_norm_fast(input, running_mean, running_var, weight, bias, training, momentum, eps)
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_sigmoid_batch_norm():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(10, 5, device='cuda')
    running_mean = torch.zeros(5, device='cuda')
    running_var = torch.ones(5, device='cuda')
    results["test_case_1"] = sigmoid_batch_norm(input_tensor, running_mean, running_var)

    # Test case 2: With learnable parameters (weight and bias)
    weight = torch.ones(5, device='cuda') * 0.5
    bias = torch.zeros(5, device='cuda') + 0.1
    results["test_case_2"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, weight=weight, bias=bias)

    # Test case 3: In training mode
    results["test_case_3"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, training=True)

    # Test case 4: With a different momentum and eps
    results["test_case_4"] = sigmoid_batch_norm(input_tensor, running_mean, running_var, momentum=0.2, eps=1e-3)

    return results

test_results = test_sigmoid_batch_norm()
