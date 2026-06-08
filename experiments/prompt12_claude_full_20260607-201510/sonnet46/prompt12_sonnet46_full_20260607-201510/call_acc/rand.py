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


def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided,
         device=None, requires_grad=False, pin_memory=False) -> Tensor:
    # Normalize size: handle both rand(3, 4) and rand([3, 4]) or rand((3, 4))
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    else:
        size = tuple(size)

    y = torch.rand(
        size,
        generator=generator,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=False,
        pin_memory=pin_memory,
    )

    if requires_grad and y.is_floating_point():
        y.requires_grad_(True)

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
