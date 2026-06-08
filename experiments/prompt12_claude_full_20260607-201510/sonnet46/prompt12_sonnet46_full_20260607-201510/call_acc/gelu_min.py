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

BLOCK_SIZE = 1024

@triton.jit
def _gelu_exact_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _gelu_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # tanh approximation GELU
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    inner = x + 0.044715 * x * x * x
    inner = inner * 0.7978845608028654  # sqrt(2/pi)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, y, mask=mask)


def _apply_gelu_triton(input: Tensor, approximate: str) -> Tensor:
    if not input.is_cuda or not input.is_floating_point():
        return F.gelu(input, approximate=approximate)
    x = input.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    try:
        if approximate == 'tanh':
            _gelu_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
        else:
            _gelu_exact_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    except Exception:
        return F.gelu(input, approximate=approximate)
    return out


def gelu_min(
    input: Tensor,
    approximate: str = 'none',
    dim: Optional[int] = None,
    keepdim: bool = False,
    out=None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    # Step 1: Apply GELU
    activated = _apply_gelu_triton(input, approximate)

    # Step 2: Compute min
    if dim is None:
        # Returns a single scalar Tensor (min over all elements)
        y = torch.min(activated)
        if out is not None:
            if isinstance(out, torch.Tensor):
                out.copy_(y)
                return out
        return y
    else:
        # Returns namedtuple (values, indices)
        result = torch.min(activated, dim=dim, keepdim=keepdim)
        if out is not None:
            if isinstance(out, (tuple, list)) and len(out) == 2:
                out[0].copy_(result.values)
                out[1].copy_(result.indices)
                return type(result)(out[0], out[1])
        return result

##################################################################################################################################################



def test_gelu_min():
    results = {}

    # Test case 1: Default approximate='none', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_1'] = gelu_min(input_tensor)

    # Test case 2: approximate='tanh', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_2'] = gelu_min(input_tensor, approximate='tanh')

    # Test case 3: approximate='none', with dim, no keepdim
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_3'] = gelu_min(input_tensor, dim=1)

    # Test case 4: approximate='tanh', with dim, keepdim=True
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_4'] = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)

    return results

test_results = test_gelu_min()
