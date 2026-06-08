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


def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    """
    Wrapper for torch.amp.autocast that mirrors the deprecated
    torch.cuda.amp.autocast interface.

    Args:
        device_type: Device type string, e.g. "cuda" or "cpu".
        enabled: Whether autocasting should be enabled. Default: True.
        dtype: Data type to use for autocasting. If None, uses the default
               for the device type (float16 for CUDA, bfloat16 for CPU).
        cache_enabled: Whether the weight caching should be enabled for
                       autocasting. Default: True.

    Returns:
        A context manager that enables mixed precision computation.
    """
    return torch.amp.autocast(
        device_type=device_type,
        enabled=enabled,
        dtype=dtype,
        cache_enabled=cache_enabled,
    )

##################################################################################################################################################



import torch

def test_autocast():
    results = {}

    # Test case 1: Basic usage with 'cuda' device type
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_1'] = tensor * 2

    # Test case 2: Explicitly disabling autocast
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_2'] = tensor * 2

    # Test case 3: Using cache_enabled set to False
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_3'] = tensor * 2

    # Test case 4: Using cache_enabled set to True
    with autocast('cuda'):
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        results['test_case_4'] = tensor * 2

    return results

test_results = test_autocast()
