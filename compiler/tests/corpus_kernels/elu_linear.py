import triton
import triton.language as tl


@triton.jit
def _elu_linear_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    alpha,
    n_rows,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    row_start = row_idx * BLOCK_SIZE
    col_start = col_idx * BLOCK_SIZE
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    
    for i in tl.range(BLOCK_SIZE):
        row = row_start + i
        if row >= n_rows:
            break
        
        for j in tl.range(BLOCK_SIZE):
            col = col_start + j
            if col >= n_cols:
                break
            
            acc = tl.zeros((), dtype=tl.float32)
            for k in tl.range(0, n_cols):
                input_val = tl.load(input_ptr + row * input_stride + k)
                weight_val = tl.load(weight_ptr + col * n_cols + k)
                acc += input_val * weight_val
            
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + col)
                acc += bias_val
            
            elu_val = tl.where(
                acc > 0.0,
                acc,
                alpha * (tl.exp(acc) - 1.0)
            )
            
            tl.store(output_ptr + row * output_stride + col, elu_val)

