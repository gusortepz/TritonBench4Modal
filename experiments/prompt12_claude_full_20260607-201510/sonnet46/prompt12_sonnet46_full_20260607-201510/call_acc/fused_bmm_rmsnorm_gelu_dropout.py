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
def _rmsnorm_gelu_kernel(
    X_ptr, Y_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    approximate_tanh: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X_ptr + row * N
    Y_ptr = Y_ptr + row * N

    # Compute sum of squares for RMS
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x * x

    mean_sq = tl.sum(acc, axis=0) / N
    rms = tl.rsqrt(mean_sq + eps)

    # Normalize and apply GELU
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = x * rms

        if approximate_tanh:
            # GELU tanh approximation
            x3 = x_norm * x_norm * x_norm
            inner = 0.7978845608 * (x_norm + 0.044715 * x3)
            tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
            y = 0.5 * x_norm * (1.0 + tanh_inner)
        else:
            # GELU exact
            y = 0.5 * x_norm * (1.0 + tl.erf(x_norm * 0.7071067811865476))

        tl.store(Y_ptr + cols, y.to(tl.float32), mask=mask)


def _rmsnorm_gelu_triton(x: Tensor, norm_size: int, eps: float, approximate: str) -> Tensor:
    """Apply RMS normalization + GELU using Triton kernel."""
    # x is (rows, norm_size)
    orig_shape = x.shape
    x_2d = x.reshape(-1, norm_size).contiguous()
    rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    BLOCK_N = min(triton.next_power_of_2(norm_size), 1024)
    approximate_tanh = (approximate == 'tanh')

    _rmsnorm_gelu_kernel[(rows,)](
        x_2d, y,
        N=norm_size,
        eps=eps,
        BLOCK_N=BLOCK_N,
        approximate_tanh=approximate_tanh,
    )
    return y.reshape(orig_shape)


def _rmsnorm_gelu_pytorch(x: Tensor, norm_size: int, eps: float, approximate: str) -> Tensor:
    """PyTorch fallback for RMS norm + GELU."""
    x_fp = x.float()
    rms = torch.rsqrt(x_fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = (x_fp * rms).to(x.dtype)
    return F.gelu(x_norm, approximate=approximate)


def fused_bmm_rmsnorm_gelu_dropout(
    input1: Tensor,
    input2: Tensor,
    normalized_shape,
    dropout_p: float = 0.1,
    eps: float = 1e-5,
    training: bool = True,
    approximate: str = 'none',
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Step 1: Batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)  # (B, N, P)

    # Determine norm_size from normalized_shape
    if isinstance(normalized_shape, int):
        norm_size = normalized_shape
    else:
        norm_size = 1
        for s in normalized_shape:
            norm_size *= s

    # Step 2 & 3: RMS Norm + GELU (fused via Triton if possible)
    use_triton = (
        bmm_out.is_cuda
        and bmm_out.is_contiguous()
        and bmm_out.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and norm_size <= bmm_out.shape[-1]
        and bmm_out.numel() > 0
    )

    if use_triton:
        try:
            y = _rmsnorm_gelu_triton(bmm_out, norm_size, eps, approximate)
        except Exception:
            y = _rmsnorm_gelu_pytorch(bmm_out, norm_size, eps, approximate)
    else:
        y = _rmsnorm_gelu_pytorch(bmm_out, norm_size, eps, approximate)

    # Step 4: Dropout
    y = F.dropout(y, p=dropout_p, training=training, inplace=False)

    if out is not None:
        out.copy_(y)
        return out
    return y

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-05, training=True, approximate='none', *, out=None):
#     z1 = torch.bmm(input1, input2)
#     rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
#     gelu_out = F.gelu(rms_norm, approximate=approximate)
#     output = F.dropout(gelu_out, p=dropout_p, training=training)
#     if out is not None:
#         out.copy_(output)
#         return out
#     return output

def test_fused_bmm_rmsnorm_gelu_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5)
    
    # Test case 2: Different dropout probability
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, dropout_p=0.2)
    
    # Test case 3: Non-training mode
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, training=False)
    
    # Test case 4: Different approximation method for GELU
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, approximate='tanh')
    
    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout()
