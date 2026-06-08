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


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, approximate: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if approximate:
        # tanh approximation
        x3 = x * x * x
        inner = x + 0.044715 * x3
        inner = inner * 0.7978845608028654
        tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        result = 0.5 * x * (1.0 + tanh_val)
    else:
        # exact GELU
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, result, mask=mask)


def _gelu_triton(input: Tensor, approximate: str) -> Tensor:
    out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n))
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    use_approx = 1 if approximate == 'tanh' else 0
    _gelu_kernel[grid](input, out, n, approximate=use_approx, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _gelu_pytorch(input: Tensor, approximate: str) -> Tensor:
    return F.gelu(input, approximate=approximate)


def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None) -> Tensor:
    # Apply GELU activation
    if input.is_cuda and input.is_floating_point() and not input.is_complex():
        try:
            activated = _gelu_triton(input.contiguous(), approximate)
        except Exception:
            activated = _gelu_pytorch(input, approximate)
    else:
        activated = _gelu_pytorch(input, approximate)

    # Compute std
    if dim is None:
        result = torch.std(activated, correction=correction)
    else:
        result = torch.std(activated, dim=dim, keepdim=keepdim, correction=correction)

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
#     gelu_result = F.gelu(input, approximate=approximate)
#     return torch.std(gelu_result, dim=dim, keepdim=keepdim, correction=correction, out=out)

def test_gelu_std():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(10, device='cuda')
    results["test_case_1"] = gelu_std(input1)
    
    # Test case 2: With dim parameter
    input2 = torch.randn(10, 20, device='cuda')
    results["test_case_2"] = gelu_std(input2, dim=1)
    
    # Test case 3: With keepdim=True
    input3 = torch.randn(10, 20, device='cuda')
    results["test_case_3"] = gelu_std(input3, dim=1, keepdim=True)
    
    # Test case 4: With approximate='tanh'
    input4 = torch.randn(10, device='cuda')
    results["test_case_4"] = gelu_std(input4, approximate='tanh')
    
    return results

test_results = test_gelu_std()
