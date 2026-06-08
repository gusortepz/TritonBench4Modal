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


def logspace(
    start,
    end,
    steps: int,
    base: float = 10.0,
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    y = torch.logspace(
        start,
        end,
        steps,
        base=base,
        out=None,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch

def test_logspace():
    results = {}

    # Test case 1: Basic functionality with default base (10.0)
    start = torch.tensor(1.0, device='cuda')
    end = torch.tensor(3.0, device='cuda')
    steps = 5
    results["test_case_1"] = logspace(start, end, steps)

    # Test case 2: Custom base (2.0)
    start = torch.tensor(0.0, device='cuda')
    end = torch.tensor(4.0, device='cuda')
    steps = 5
    base = 2.0
    results["test_case_2"] = logspace(start, end, steps, base=base)

    # Test case 3: Custom dtype (float64)
    start = torch.tensor(1.0, device='cuda')
    end = torch.tensor(2.0, device='cuda')
    steps = 4
    dtype = torch.float64
    results["test_case_3"] = logspace(start, end, steps, dtype=dtype)

    # Test case 4: Requires gradient
    start = torch.tensor(1.0, device='cuda')
    end = torch.tensor(3.0, device='cuda')
    steps = 3
    requires_grad = True
    results["test_case_4"] = logspace(start, end, steps, requires_grad=requires_grad)

    return results

test_results = test_logspace()
