import torch
import triton
import triton.language as tl
from task import input_t, output_t
from utils import DeterministicContext

"""
Atomic Add Approach

Uses atomic operations to accumulate the sum directly.
Simpler implementation but may have contention issues.
"""

# Tunable parameters
BLOCK_SIZE = 2048


@triton.jit
def sum_reduction_kernel_atomic(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    data_f64 = data.to(tl.float64)
    block_sum = tl.sum(data_f64, axis=0)

    # Atomic add to global output
    tl.atomic_add(output_ptr, block_sum)


def custom_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        data, output = data

        data_flat = data.flatten()
        n_elements = data_flat.numel()

        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

        # Initialize output
        output_tensor = torch.zeros(1, dtype=torch.float64, device=data.device)

        grid = (n_blocks,)
        sum_reduction_kernel_atomic[grid](
            data_flat,
            output_tensor,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output_tensor[0].to(torch.float32)
