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


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
) -> Tensor:
    """
    Computes output using input values and pixel locations from grid.
    Supports spatial (4-D) and volumetric (5-D) input.
    Interpolates output value at specified grid positions using nearest or bilinear interpolation.
    Grid values are normalized within [-1, 1] range.
    Values outside are handled by padding_mode.
    Often used with affine_grid to build Spatial Transformer Networks.
    """
    return F.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_grid_sample():
    results = {}

    # Test case 1: 4D input, bilinear mode, zeros padding
    input_4d = torch.rand(1, 3, 4, 4, device='cuda')
    grid_4d = torch.rand(1, 2, 2, 2, device='cuda') * 2 - 1  # Range [-1, 1]
    results["test_case_1"] = grid_sample(input_4d, grid_4d)

    # Test case 2: 4D input, nearest mode, border padding
    results["test_case_2"] = grid_sample(input_4d, grid_4d, mode='nearest', padding_mode='border')

    # Test case 3: 5D input, bilinear mode, reflection padding
    input_5d = torch.rand(1, 3, 4, 4, 4, device='cuda')
    grid_5d = torch.rand(1, 2, 2, 2, 3, device='cuda') * 2 - 1  # Range [-1, 1]
    results["test_case_3"] = grid_sample(input_5d, grid_5d, padding_mode='reflection')

    # Test case 4: 5D input, nearest mode, zeros padding, align_corners=True
    results["test_case_4"] = grid_sample(input_5d, grid_5d, mode='nearest', align_corners=True)

    return results

test_results = test_grid_sample()
