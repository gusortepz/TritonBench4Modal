import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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
def _relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Applies the rectified linear unit function element-wise.
    
    Args:
        input: input tensor
        inplace: if True, performs the operation in-place. Default: False
        
    Returns:
        output tensor with ReLU applied
    """
    if not input.is_cuda or input.dtype not in (torch.float32, torch.float64):
        return F.relu(input, inplace=inplace)
    
    if inplace:
        return F.relu(input, inplace=True)
    
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    if n_elements == 0:
        return output
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    
    _relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu(input: torch.Tensor, inplace: bool=False) -> torch.Tensor:
#     return F.relu(input, inplace=inplace)

def test_relu():
    results = {}
    
    # Test case 1: Basic test with a simple tensor
    input1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = relu(input1)
    
    # Test case 2: Test with a 2D tensor
    input2 = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device='cuda')
    results["test_case_2"] = relu(input2)
    
    # Test case 3: Test with inplace=True
    input3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = relu(input3, inplace=True)
    
    # Test case 4: Test with a larger tensor
    input4 = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], device='cuda')
    results["test_case_4"] = relu(input4)
    
    return results

test_results = test_relu()
