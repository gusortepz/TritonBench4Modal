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


def ones_like(
    input: Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Tensor:
    """
    Returns a tensor filled with the scalar value 1, with the same size as the input tensor.
    Mirrors the properties of the input in terms of dtype, layout, device, and memory format
    unless specified otherwise.
    """
    return torch.ones_like(
        input,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        memory_format=memory_format,
    )

##################################################################################################################################################



import torch

def test_ones_like():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_1"] = ones_like(input_tensor)

    # Test case 2: Test with a different dtype
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_2"] = ones_like(input_tensor, dtype=torch.float64)

    # Test case 3: Test with requires_grad=True
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_3"] = ones_like(input_tensor, requires_grad=True)

    # Test case 4: Test with a different device
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_4"] = ones_like(input_tensor, device='cuda')

    return results

test_results = test_ones_like()
