import triton
import triton.language as tl


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


