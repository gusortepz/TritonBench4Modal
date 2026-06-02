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
def _relu_sqrt_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    z = tl.sqrt(tl.maximum(x, 0.0))
    tl.store(out_ptr + offs, z, mask=mask)

def relu_sqrt(input: Tensor, inplace: bool = False, out: Optional[Tensor] = None) -> Tensor:
    def ref():
        r = torch.sqrt(F.relu(input))
        if out is not None:
            out.copy_(r)
            return out
        return r

    if not isinstance(input, torch.Tensor):
        return ref()
    if not input.is_cuda or input.is_complex():
        return ref()
    if not input.is_floating_point():
        return ref()
    if out is not None and (not isinstance(out, torch.Tensor) or out.shape != input.shape or out.dtype != input.dtype):
        return ref()

    x_c = input.contiguous()
    out_t = out if out is not None and out.is_cuda and out.is_contiguous() else torch.empty_like(x_c)
    n = x_c.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _relu_sqrt_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
    
    if out is not None and out_t is not out:
        out.copy_(out_t)
        return out
    return out_t

##################################################################################################################################################



import torch
from torch import Tensor

# def relu_sqrt(input: Tensor, inplace: bool=False, out: Tensor=None) -> Tensor:
#     if input.dtype != torch.float32 and input.dtype != torch.float64:
#         input = input.float()
#     if inplace:
#         input.relu_()
#         input.sqrt_()
#         return input
#     elif out is not None:
#         out.copy_(torch.sqrt(torch.relu(input)))
#         return out
#     else:
#         return torch.sqrt(torch.relu(input))

def test_relu_sqrt():
    results = {}
    
    # Test case 1: Default parameters
    a = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = relu_sqrt(a)
    
    # Test case 2: Inplace operation
    b = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_2"] = relu_sqrt(b, inplace=True)
    
    # Test case 3: Out parameter
    c = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    out = torch.empty_like(c)
    results["test_case_3"] = relu_sqrt(c, out=out)
    
    # Test case 4: Non-float input
    d = torch.tensor([-1, 0, 4, 9], device='cuda')
    results["test_case_4"] = relu_sqrt(d)
    
    return results

test_results = test_relu_sqrt()
