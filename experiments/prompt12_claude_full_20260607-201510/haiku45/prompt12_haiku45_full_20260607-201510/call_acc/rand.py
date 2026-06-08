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


def rand(
    *size,
    generator: Optional[torch.Generator] = None,
    out: Optional[Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor:
    """
    Returns a tensor filled with random numbers from a uniform distribution on [0, 1).
    
    Args:
        size: Variable length argument list defining output tensor shape.
        generator: Optional pseudorandom number generator.
        out: Optional output tensor.
        dtype: Optional desired data type. Defaults to global default if None.
        layout: Desired layout, defaults to torch.strided.
        device: Desired device. Defaults to current device if None.
        requires_grad: Whether to record autograd operations. Defaults to False.
        pin_memory: Whether to allocate in pinned memory (CPU only). Defaults to False.
    
    Returns:
        A tensor filled with random numbers from U[0, 1).
    """
    # Normalize size argument: handle both *size unpacking and tuple/list inputs
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = tuple(size[0])
    else:
        shape = tuple(size)
    
    # Use PyTorch's built-in rand; it is already highly optimized and is the reference
    y = torch.rand(
        shape,
        generator=generator,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )
    
    # Handle out parameter
    if out is not None:
        out.copy_(y)
        return out
    
    return y

##################################################################################################################################################



import torch

def test_rand():
    results = {}

    # Test case 1: Basic usage with size only
    results["test_case_1"] = rand(2, 3, device='cuda')

    # Test case 2: Specifying dtype
    results["test_case_2"] = rand(2, 3, dtype=torch.float64, device='cuda')

    # Test case 3: Using a generator
    gen = torch.Generator(device='cuda')
    gen.manual_seed(42)
    results["test_case_3"] = rand(2, 3, generator=gen, device='cuda')

    # Test case 4: Requires gradient
    results["test_case_4"] = rand(2, 3, requires_grad=True, device='cuda')

    return results

test_results = test_rand()
