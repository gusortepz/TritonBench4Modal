import torch
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@triton.jit
def _tanh_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    tl.store(out_ptr + offs, y, mask=mask)

def tanh(input, *, out=None):
    if isinstance(input, torch.Tensor) and input.is_cuda and input.is_floating_point():
        x_c = input.contiguous()
        out_t = out if isinstance(out, torch.Tensor) and out.is_cuda and out.shape == x_c.shape and out.dtype == x_c.dtype and out.is_contiguous() else torch.empty_like(x_c)
        n = x_c.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _tanh_kernel[grid](x_c, out_t, n, BLOCK_SIZE=1024)
        if out is not None and out_t is not out:
            out.copy_(out_t)
            return out
        return out_t
    r = torch.tanh(input)
    if out is not None:
        out.copy_(r)
        return out
    return r

##################################################################################################################################################



import torch

def test_tanh():
    results = {}

    # Test case 1: Basic test with a simple tensor
    input_tensor_1 = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5], device='cuda')
    results["test_case_1"] = tanh(input_tensor_1)

    # Test case 2: Test with a 2D tensor
    input_tensor_2 = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], device='cuda')
    results["test_case_2"] = tanh(input_tensor_2)

    # Test case 3: Test with a larger tensor
    input_tensor_3 = torch.randn(100, 100, device='cuda')
    results["test_case_3"] = tanh(input_tensor_3)

    # Test case 4: Test with an empty tensor
    input_tensor_4 = torch.tensor([], device='cuda')
    results["test_case_4"] = tanh(input_tensor_4)

    return results

test_results = test_tanh()
