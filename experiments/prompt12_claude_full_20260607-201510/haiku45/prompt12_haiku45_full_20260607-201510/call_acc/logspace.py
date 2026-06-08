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
    start: Union[float, Tensor],
    end: Union[float, Tensor],
    steps: int,
    base: float = 10.0,
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Creates a one-dimensional tensor of size 'steps' whose values are evenly spaced
    from base^start to base^end, inclusive, on a logarithmic scale with a specified base.
    
    Args:
        start (float or Tensor): the starting value for the set of points.
        end (float or Tensor): the ending value for the set of points.
        steps (int): size of the constructed tensor
        base (float, optional): base of the logarithm function. Default: 10.0.
    
    Keyword arguments:
        out (Tensor, optional): the output tensor.
        dtype (torch.dtype, optional): the data type to perform the computation in.
        layout (torch.layout, optional): the desired layout of returned Tensor.
        device (torch.device, optional): the desired device of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
        Tensor: A tensor of size 'steps' with logarithmically spaced values.
    """
    
    # Convert scalar tensors to Python floats if needed
    if isinstance(start, Tensor):
        if start.dim() != 0:
            raise ValueError("start tensor must be 0-dimensional")
        start = start.item()
    
    if isinstance(end, Tensor):
        if end.dim() != 0:
            raise ValueError("end tensor must be 0-dimensional")
        end = end.item()
    
    # Use PyTorch's logspace directly
    y = torch.logspace(
        start,
        end,
        steps=steps,
        base=base,
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
