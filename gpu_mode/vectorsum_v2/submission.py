import torch
import triton
import triton.language as tl
from task import input_t, output_t
from utils import DeterministicContext

@triton.jit
def sum_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to sum all elements in a tensor.
    Each block reduces BLOCK_SIZE elements to a single value.
    """
    # Get the block ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements (handles non-multiple of BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data (masked load returns 0 for out-of-bounds)
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float64 for precision
    data_f64 = data.to(tl.float64)
    
    # Sum within this block
    block_sum = tl.sum(data_f64, axis=0)
    
    # Atomic add to global output (all blocks contribute)
    tl.atomic_add(output_ptr, block_sum)


def custom_kernel(data: input_t) -> output_t:
    print("-"*10)
    print("running the sum reduction KERNEL")
    print("-"*10)
    with DeterministicContext():
        data, output = data
        
        # Flatten the input tensor
        data_flat = data.flatten()
        n_elements = data_flat.numel()
        
        # Allocate output tensor (scalar) in float64 for accumulation
        output_temp = torch.zeros(1, dtype=torch.float64, device=data.device)
        
        # Choose block size (tune this for your GPU)
        BLOCK_SIZE = 1024
        
        # Calculate grid size (number of blocks)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        sum_reduction_kernel[grid](
            data_flat,
            output_temp,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Convert back to float32
        output = output_temp.to(torch.float32).squeeze()
        
        return output