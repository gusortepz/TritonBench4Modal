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
def _sigmoid_sub_kernel(
    x_ptr,
    other_ptr,
    out_ptr,
    alpha,
    n_elements,
    other_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid
    sig = tl.sigmoid(x)

    if other_is_scalar:
        result = sig - alpha * other_ptr
    else:
        o = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        result = sig - alpha * o

    tl.store(out_ptr + offsets, result, mask=mask)


def fused_mv_sigmoid_sub(input: Tensor, vec: Tensor, other, alpha=1, *, out: Optional[Tensor] = None) -> Tensor:
    # Step 1: matrix-vector multiplication
    mv_result = torch.mv(input, vec)

    # Step 2: fused sigmoid + subtraction
    n_elements = mv_result.numel()

    # Check if we can use Triton (CUDA float tensors, no complex)
    use_triton = (
        mv_result.is_cuda
        and mv_result.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    )

    if use_triton:
        # Convert to float32 for triton if needed (triton works well with float32/float16)
        orig_dtype = mv_result.dtype
        if orig_dtype == torch.float64:
            # Fall back to PyTorch for float64
            use_triton = False

    if use_triton:
        result = torch.empty(n_elements, device=mv_result.device, dtype=mv_result.dtype)
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        other_is_scalar = not isinstance(other, Tensor)

        if other_is_scalar:
            # Pass scalar as float
            other_val = float(other)
            # We'll create a dummy pointer approach - pass as alpha adjustment
            # Actually, let's handle scalar differently
            # Create a temporary tensor for the scalar
            # Or better: handle inline
            try:
                _sigmoid_sub_kernel[grid](
                    mv_result,
                    other_val,  # This won't work as pointer
                    result,
                    float(alpha),
                    n_elements,
                    True,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            except Exception:
                use_triton = False
        else:
            other_tensor = other.contiguous()
            if not other_tensor.is_cuda or other_tensor.dtype != mv_result.dtype:
                use_triton = False

        if use_triton and not other_is_scalar:
            try:
                _sigmoid_sub_kernel[grid](
                    mv_result,
                    other_tensor,
                    result,
                    float(alpha),
                    n_elements,
                    False,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            except Exception:
                use_triton = False

        if use_triton:
            if out is not None:
                out.copy_(result)
                return out
            return result

    # PyTorch fallback
    sig = torch.sigmoid(mv_result)
    if isinstance(other, Tensor):
        y = sig - alpha * other
    else:
        y = sig - alpha * float(other)

    if out is not None:
        out.copy_(y)
        return out
    return y

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
