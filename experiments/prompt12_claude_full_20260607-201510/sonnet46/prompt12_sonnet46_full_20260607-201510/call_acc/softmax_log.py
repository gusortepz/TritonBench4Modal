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


def _softmax_log_impl(input: Tensor, dim: int = -1, dtype=None) -> Tensor:
    x = input
    if dtype is not None:
        x = x.to(dtype)
    # softmax(log(x)) = exp(log(x)) / sum(exp(log(x))) = x / sum(x)
    # This is just L1 normalization along the given dim (for positive inputs)
    # But to be safe and general (handles non-positive via log then softmax):
    log_x = torch.log(x)
    result = F.softmax(log_x, dim=dim)
    return result


try:
    _softmax_log_fast = torch.compile(_softmax_log_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _softmax_log_fast = _softmax_log_impl


def softmax_log(input, dim=-1, dtype=None) -> Tensor:
    try:
        return _softmax_log_fast(input, dim=dim, dtype=dtype)
    except Exception:
        return _softmax_log_impl(input, dim=dim, dtype=dtype)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax_log(input, dim=-1, dtype=None):
#     if dtype is not None:
#         input = input.to(dtype)
#     log_input = input.log()
#     return F.softmax(log_input, dim=dim)

def test_softmax_log():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = softmax_log(input_tensor)

    # Test case 2: Specifying a different dimension
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = softmax_log(input_tensor, dim=0)

    # Test case 3: Specifying a different dtype
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = softmax_log(input_tensor, dtype=torch.float64)

    # Test case 4: Larger tensor
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_4"] = softmax_log(input_tensor)

    return results

test_results = test_softmax_log()
