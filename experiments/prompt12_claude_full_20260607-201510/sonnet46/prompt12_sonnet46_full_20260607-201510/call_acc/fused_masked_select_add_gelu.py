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
def _add_gelu_kernel(
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

    if other_is_scalar:
        other_val = other_ptr  # treated as float constant via meta
        val = x + alpha * other_val
    else:
        o = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        val = x + alpha * o

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_out = 0.5 * val * (1.0 + tl.erf(val * 0.7071067811865476))

    tl.store(out_ptr + offsets, gelu_out, mask=mask)


@triton.jit
def _add_gelu_tanh_kernel(
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

    if other_is_scalar:
        other_val = other_ptr
        val = x + alpha * other_val
    else:
        o = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        val = x + alpha * o

    # GELU tanh approximation
    inner = val + 0.044715 * val * val * val
    inner = inner * 0.7978845608028654
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu_out = 0.5 * val * (1.0 + tanh_val)

    tl.store(out_ptr + offsets, gelu_out, mask=mask)


def _add_gelu_triton(selected: Tensor, other, alpha: float, approximate: str) -> Tensor:
    n = selected.numel()
    if n == 0:
        return selected.clone()

    # Ensure float
    if selected.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        selected = selected.float()

    out = torch.empty_like(selected)
    BLOCK_SIZE = min(triton.next_power_of_2(n), 1024)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    if isinstance(other, (int, float)):
        other_is_scalar = True
        other_val = float(other)
        # Pass scalar as a dummy pointer trick won't work cleanly;
        # we'll store it as a tensor
        other_tensor = torch.tensor(other_val, dtype=selected.dtype, device=selected.device)
        if approximate == 'tanh':
            _add_gelu_tanh_kernel[grid](
                selected, other_tensor, out,
                float(alpha), n,
                True,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _add_gelu_kernel[grid](
                selected, other_tensor, out,
                float(alpha), n,
                True,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        # other is a tensor - need to handle shape
        other_t = other.to(dtype=selected.dtype, device=selected.device)
        # other_t must be broadcastable to selected; flatten for elementwise
        if other_t.numel() == 1:
            # treat as scalar-like: expand
            other_t = other_t.expand(n).contiguous()
        else:
            other_t = other_t.flatten()
            if other_t.numel() != n:
                # fallback
                return _add_gelu_fallback(selected, other, alpha, approximate)
            other_t = other_t.contiguous()

        if approximate == 'tanh':
            _add_gelu_tanh_kernel[grid](
                selected, other_t, out,
                float(alpha), n,
                False,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _add_gelu_kernel[grid](
                selected, other_t, out,
                float(alpha), n,
                False,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out


def _add_gelu_fallback(selected: Tensor, other, alpha, approximate: str) -> Tensor:
    if isinstance(other, Tensor):
        val = selected + alpha * other.flatten()[:selected.numel()]
    else:
        val = selected + alpha * other
    return F.gelu(val, approximate=approximate)


def fused_masked_select_add_gelu(
    input: Tensor,
    mask: Tensor,
    other,
    *,
    alpha=1,
    approximate: str = 'none',
    out: Optional[Tensor] = None,
) -> Tensor:
    # Step 1: masked select - returns 1D tensor of selected elements
    selected = torch.masked_select(input, mask)

    # Step 2 & 3: add other*alpha then GELU
    if selected.is_cuda and selected.dtype in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64
    ):
        try:
            result = _add_gelu_triton(selected, other, float(alpha), approximate)
        except Exception:
            result = _add_gelu_fallback(selected, other, float(alpha), approximate)
    else:
        result = _add_gelu_fallback(selected, other, float(alpha), approximate)

    if out is not None:
        out.copy_(result)
        return out
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F


def test_fused_masked_select_add_gelu():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask1 = torch.tensor([True, False, True, False], device='cuda')
    other1 = 1.0
    results["test_case_1"] = fused_masked_select_add_gelu(input1, mask1, other1)
    
    # Test case 2: Test with alpha parameter
    input2 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask2 = torch.tensor([True, True, False, False], device='cuda')
    other2 = 2.0
    results["test_case_2"] = fused_masked_select_add_gelu(input2, mask2, other2, alpha=0.5)
    
    # Test case 3: Test with approximate='tanh'
    input3 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask3 = torch.tensor([False, True, True, False], device='cuda')
    other3 = 1.0
    results["test_case_3"] = fused_masked_select_add_gelu(input3, mask3, other3, approximate='tanh')
    
    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask4 = torch.tensor([True, False, True, True], device='cuda')
    other4 = 1.0
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = fused_masked_select_add_gelu(input4, mask4, other4, out=out4)
    
    return results

test_results = test_fused_masked_select_add_gelu()
