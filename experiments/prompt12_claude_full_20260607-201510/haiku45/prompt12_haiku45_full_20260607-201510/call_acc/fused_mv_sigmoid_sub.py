import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Union
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
def _fused_mv_sigmoid_sub_kernel(
    output_ptr,
    input_ptr,
    vec_ptr,
    other_ptr,
    alpha,
    n: tl.constexpr,
    m: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    is_other_scalar: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n)
    
    for i in tl.range(block_start, block_end):
        acc = 0.0
        for j in tl.range(0, m):
            input_val = tl.load(input_ptr + i * m + j)
            vec_val = tl.load(vec_ptr + j)
            acc += input_val * vec_val
        
        sigmoid_val = tl.sigmoid(acc)
        
        if is_other_scalar:
            other_val = tl.load(other_ptr)
        else:
            other_val = tl.load(other_ptr + i)
        
        result = sigmoid_val - alpha * other_val
        tl.store(output_ptr + i, result)


def fused_mv_sigmoid_sub(
    input: Tensor,
    vec: Tensor,
    other: Union[Tensor, float],
    alpha: float = 1,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    if not input.is_cuda or not vec.is_cuda:
        y = _fused_mv_sigmoid_sub_pytorch(input, vec, other, alpha)
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    if isinstance(other, Tensor) and not other.is_cuda:
        y = _fused_mv_sigmoid_sub_pytorch(input, vec, other, alpha)
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    n, m = input.shape
    device = input.device
    dtype = input.dtype
    
    if dtype not in (torch.float32, torch.float64):
        y = _fused_mv_sigmoid_sub_pytorch(input, vec, other, alpha)
        if out is not None:
            out.copy_(y)
            return out
        return y
    
    output = torch.empty(n, device=device, dtype=dtype)
    
    is_other_scalar = not isinstance(other, Tensor)
    if is_other_scalar:
        other_tensor = torch.tensor(float(other), device=device, dtype=dtype)
    else:
        other_tensor = other
    
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    try:
        _fused_mv_sigmoid_sub_kernel[grid](
            output,
            input,
            vec,
            other_tensor,
            float(alpha),
            n,
            m,
            BLOCK_SIZE=BLOCK_SIZE,
            is_other_scalar=is_other_scalar,
        )
    except Exception:
        return _fused_mv_sigmoid_sub_pytorch(input, vec, other, alpha)
    
    if out is not None:
        out.copy_(output)
        return out
    return output


def _fused_mv_sigmoid_sub_pytorch(
    input: Tensor,
    vec: Tensor,
    other: Union[Tensor, float],
    alpha: float = 1,
) -> Tensor:
    mv_result = torch.mv(input, vec)
    sigmoid_result = torch.sigmoid(mv_result)
    
    if isinstance(other, Tensor):
        result = sigmoid_result - alpha * other
    else:
        result = sigmoid_result - alpha * float(other)
    
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
#     """
#     Performs a fused operation combining matrix-vector multiplication, sigmoid activation, and subtraction.

#     Args:
#         input (Tensor): Input matrix A of shape (n, m).
#         vec (Tensor): Input vector v of shape (m).
#         other (Tensor or Number): Tensor or scalar b to subtract from the sigmoid output, scaled by alpha.
#         alpha (Number, optional): Scalar multiplier for other. Default: 1.
#         out (Tensor, optional): Output tensor. Ignored if None. Default: None.

#     Returns:
#         Tensor: The result of the fused operation.
#     """
#     z = torch.mv(input, vec)
#     s = torch.sigmoid(z)
#     y = torch.sub(s, other, alpha=alpha)
#     if out is not None:
#         out.copy_(y)
#         return out
#     return y

def test_fused_mv_sigmoid_sub():
    results = {}
    
    # Test case 1: Basic functionality
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec1 = torch.tensor([1.0, 1.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5], device='cuda')
    results["test_case_1"] = fused_mv_sigmoid_sub(input1, vec1, other1)
    
    # Test case 2: Scalar other
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec2 = torch.tensor([1.0, 1.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = fused_mv_sigmoid_sub(input2, vec2, other2)
    
    # Test case 3: Different alpha
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec3 = torch.tensor([1.0, 1.0], device='cuda')
    other3 = torch.tensor([0.5, 0.5], device='cuda')
    results["test_case_3"] = fused_mv_sigmoid_sub(input3, vec3, other3, alpha=2)
    
    # Test case 4: Output tensor provided
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec4 = torch.tensor([1.0, 1.0], device='cuda')
    other4 = torch.tensor([0.5, 0.5], device='cuda')
    out4 = torch.empty(2, device='cuda')
    results["test_case_4"] = fused_mv_sigmoid_sub(input4, vec4, other4, out=out4)
    
    return results

test_results = test_fused_mv_sigmoid_sub()
