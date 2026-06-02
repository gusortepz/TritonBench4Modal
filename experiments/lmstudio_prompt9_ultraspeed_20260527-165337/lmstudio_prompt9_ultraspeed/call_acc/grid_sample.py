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

@triton.jit
def _grid_sample_kernel(
    input_ptr, grid_ptr, output_ptr,
    batch_size, channels, height, width, depth,
    mode, padding_mode, align_corners,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (channels * height * width * depth)
    rest = pid % (channels * height * width * depth)
    channel_idx = rest // (height * width * depth)
    rest = rest % (height * width * depth)
    h_idx = rest // (width * depth)
    w_idx = rest // depth
    d_idx = rest % depth

    if batch_idx >= batch_size or channel_idx >= channels or h_idx >= height or w_idx >= width or d_idx >= depth:
        return

    grid_val = tl.load(grid_ptr + batch_idx * height * width * depth * 3 + h_idx * width * depth * 3 + w_idx * depth * 3 + d_idx * 3)
    grid_x = grid_val[0]
    grid_y = grid_val[1]
    grid_z = grid_val[2]

    if align_corners:
        x = (grid_x + 1.0) * (width - 1) / 2.0
        y = (grid_y + 1.0) * (height - 1) / 2.0
        z = (grid_z + 1.0) * (depth - 1) / 2.0
    else:
        x = (grid_x + 1.0) * width / 2.0 - 0.5
        y = (grid_y + 1.0) * height / 2.0 - 0.5
        z = (grid_z + 1.0) * depth / 2.0 - 0.5

    if mode == 0:  # nearest
        x = tl.floor(x + 0.5)
        y = tl.floor(y + 0.5)
        z = tl.floor(z + 0.5)
        x = tl.clamp(x, 0.0, width - 1.0)
        y = tl.clamp(y, 0.0, height - 1.0)
        z = tl.clamp(z, 0.0, depth - 1.0)
        x_idx = tl.cast(x, tl.int32)
        y_idx = tl.cast(y, tl.int32)
        z_idx = tl.cast(z, tl.int32)
    else:  # bilinear
        x0 = tl.floor(x)
        y0 = tl.floor(y)
        z0 = tl.floor(z)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        z1 = z0 + 1.0

        x0 = tl.clamp(x0, 0.0, width - 1.0)
        y0 = tl.clamp(y0, 0.0, height - 1.0)
        z0 = tl.clamp(z0, 0.0, depth - 1.0)
        x1 = tl.clamp(x1, 0.0, width - 1.0)
        y1 = tl.clamp(y1, 0.0, height - 1.0)
        z1 = tl.clamp(z1, 0.0, depth - 1.0)

        x0 = tl.cast(x0, tl.int32)
        y0 = tl.cast(y0, tl.int32)
        z0 = tl.cast(z0, tl.int32)
        x1 = tl.cast(x1, tl.int32)
        y1 = tl.cast(y1, tl.int32)
        z1 = tl.cast(z1, tl.int32)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        # Interpolation in 3D
        val000 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y0 * width * depth + x0 * depth + z0)
        val001 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y0 * width * depth + x0 * depth + z1)
        val010 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y0 * width * depth + x1 * depth + z0)
        val011 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y0 * width * depth + x1 * depth + z1)
        val100 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y1 * width * depth + x0 * depth + z0)
        val101 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y1 * width * depth + x0 * depth + z1)
        val110 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y1 * width * depth + x1 * depth + z0)
        val111 = tl.load(input_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + y1 * width * depth + x1 * depth + z1)

        val00 = val000 * (1.0 - dx) + val010 * dx
        val01 = val001 * (1.0 - dx) + val011 * dx
        val10 = val100 * (1.0 - dx) + val110 * dx
        val11 = val101 * (1.0 - dx) + val111 * dx

        val0 = val00 * (1.0 - dy) + val10 * dy
        val1 = val01 * (1.0 - dy) + val11 * dy

        val = val0 * (1.0 - dz) + val1 * dz

    tl.store(output_ptr + batch_idx * channels * height * width * depth + channel_idx * height * width * depth + h_idx * width * depth + w_idx * depth + d_idx, val)

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    if not (input.is_cuda and grid.is_cuda):
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    if input.dim() == 4:
        batch_size, channels, height, width = input.shape
        depth = 1
    elif input.dim() == 5:
        batch_size, channels, height, width, depth = input.shape
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    if grid.dim() == 4:
        grid_h, grid_w, grid_d = grid.shape[2], grid.shape[3], 1
    elif grid.dim() == 5:
        grid_h, grid_w, grid_d = grid.shape[2], grid.shape[3], grid.shape[4]
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    if not (grid_h == height and grid_w == width and grid_d == depth):
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    if not (input.is_floating_point() and grid.is_floating_point()):
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    output = torch.empty_like(input)
    
    if mode == 'bilinear':
        mode_val = 1
    elif mode == 'nearest':
        mode_val = 0
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    if padding_mode == 'zeros':
        padding_mode_val = 0
    elif padding_mode == 'border':
        padding_mode_val = 1
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    n_elements = batch_size * channels * height * width * depth
    grid_size = triton.cdiv(n_elements, 1024)
    
    _grid_sample_kernel[grid_size](input, grid, output, batch_size, channels, height, width, depth, mode_val, padding_mode_val, align_corners, BLOCK_SIZE=1024)
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_grid_sample():
    results = {}

    # Test case 1: 4D input, bilinear mode, zeros padding
    input_4d = torch.rand(1, 3, 4, 4, device='cuda')
    grid_4d = torch.rand(1, 2, 2, 2, device='cuda') * 2 - 1  # Range [-1, 1]
    results["test_case_1"] = grid_sample(input_4d, grid_4d)

    # Test case 2: 4D input, nearest mode, border padding
    results["test_case_2"] = grid_sample(input_4d, grid_4d, mode='nearest', padding_mode='border')

    # Test case 3: 5D input, bilinear mode, reflection padding
    input_5d = torch.rand(1, 3, 4, 4, 4, device='cuda')
    grid_5d = torch.rand(1, 2, 2, 2, 3, device='cuda') * 2 - 1  # Range [-1, 1]
    results["test_case_3"] = grid_sample(input_5d, grid_5d, padding_mode='reflection')

    # Test case 4: 5D input, nearest mode, zeros padding, align_corners=True
    results["test_case_4"] = grid_sample(input_5d, grid_5d, mode='nearest', align_corners=True)

    return results

test_results = test_grid_sample()
