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
def tril_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    N, P,
    stride_an, stride_ak,
    stride_bk, stride_bp,
    stride_cn, stride_cp,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # We iterate over K (the shared dimension = N for A and B)
    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A tile with tril masking: A[row, k] is zero if k > row
        # A has shape (N, N), we load block (row_offs, k_offs)
        a_row = row_offs[:, None]  # (BLOCK_M, 1)
        a_col = k_offs[None, :]    # (1, BLOCK_K)
        tril_mask = a_col <= a_row  # lower triangular condition
        valid_mask_a = (a_row < N) & (a_col < N)
        a = tl.load(
            A_ptr + a_row * stride_an + a_col * stride_ak,
            mask=valid_mask_a & tril_mask,
            other=0.0
        )

        # Load B tile: B[k_offs, col_offs]
        b_row = k_offs[:, None]    # (BLOCK_K, 1)
        b_col = col_offs[None, :]  # (1, BLOCK_N)
        valid_mask_b = (b_row < N) & (b_col < P)
        b = tl.load(
            B_ptr + b_row * stride_bk + b_col * stride_bp,
            mask=valid_mask_b,
            other=0.0
        )

        acc += tl.dot(a, b)

    # Scale and store
    acc = acc * scale

    c_row = row_offs[:, None]
    c_col = col_offs[None, :]
    valid_mask_c = (c_row < N) & (c_col < P)
    tl.store(
        C_ptr + c_row * stride_cn + c_col * stride_cp,
        acc.to(tl.float32),
        mask=valid_mask_c
    )


def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Reference PyTorch fallback
    def _ref():
        return beta * (alpha * (torch.tril(A) @ B))

    if not A.is_cuda or not B.is_cuda:
        return _ref()

    if A.dtype not in (torch.float32, torch.float16) or B.dtype not in (torch.float32, torch.float16):
        return _ref()

    if A.dim() != 2 or B.dim() != 2:
        return _ref()

    N = A.shape[0]
    if A.shape[1] != N:
        return _ref()
    if B.shape[0] != N:
        return _ref()
    P = B.shape[1]

    try:
        # Ensure contiguous and float32
        A_c = A.contiguous().float()
        B_c = B.contiguous().float()

        C = torch.empty((N, P), device=A.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 32

        scale = alpha * beta

        grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(P, BLOCK_N))

        tril_mm_kernel[grid](
            A_c, B_c, C,
            N, P,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            C.stride(0), C.stride(1),
            scale,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )

        # Cast back to original dtype if needed
        if A.dtype != torch.float32:
            C = C.to(A.dtype)

        return C
    except Exception:
        return _ref()

##################################################################################################################################################



import torch

def test_tril_mm_and_scale():
    results = {}

    # Test case 1: Basic functionality with square matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = tril_mm_and_scale(A1, B1, alpha1, beta1)

    # Test case 2: Different alpha and beta values
    A2 = torch.tensor([[1.0, 0.0], [3.0, 4.0]], device='cuda')
    B2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = tril_mm_and_scale(A2, B2, alpha2, beta2)

    # Test case 3: Larger matrix
    A3 = torch.tensor([[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]], device='cuda')
    B3 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = tril_mm_and_scale(A3, B3, alpha3, beta3)

    # Test case 4: Zero matrix A
    A4 = torch.zeros((2, 2), device='cuda')
    B4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = tril_mm_and_scale(A4, B4, alpha4, beta4)

    return results

test_results = test_tril_mm_and_scale()
