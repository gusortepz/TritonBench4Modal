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

def sqrt(input, *, out=None):
    return torch.sqrt(input, out=out)
