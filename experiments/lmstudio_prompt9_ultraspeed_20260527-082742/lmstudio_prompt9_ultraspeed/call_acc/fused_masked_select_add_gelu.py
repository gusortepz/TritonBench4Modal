import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def _fused_masked_select_add_gelu_impl(input, mask, other, alpha, approximate):
    if isinstance(other, torch.Tensor):
        other = other.to(input.device)
    return F.gelu(input[mask] + alpha * other, approximate=approximate)

try:
    _fused_masked_select_add_gelu_fast = torch.compile(_fused_masked_select_add_gelu_impl, mode="max-autotune", fullgraph=False)
except Exception:
    _fused_masked_select_add_gelu_fast = _fused_masked_select_add_gelu_impl

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    y = _fused_masked_select_add_gelu_fast(input, mask, other, alpha, approximate)
    if out is not None:
        out.copy_(y)
        return out
    return y

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
