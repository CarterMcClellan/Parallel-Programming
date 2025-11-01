import torch
import triton
import triton.language as tl
from task import input_t, output_t
from utils import DeterministicContext

"""
Two-Stage Reduction Approach

Stage 1: Reduce to partial sums in parallel (no atomics)
Stage 2: Final reduction on CPU using PyTorch

This approach avoids atomic contention by using a hierarchical reduction.
"""

# Tunable parameters
BLOCK_SIZE = 1024


@triton.jit
def sum_reduction_kernel_stage1(
    input_ptr,
    partial_sums_ptr,
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

    # Store partial sum (no atomics!)
    tl.store(partial_sums_ptr + pid, block_sum)


def custom_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        data, output = data

        data_flat = data.flatten()
        n_elements = data_flat.numel()

        n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

        # Stage 1: Reduce to partial sums
        partial_sums = torch.empty(n_blocks, dtype=torch.float64, device=data.device)
        grid = (n_blocks,)
        sum_reduction_kernel_stage1[grid](
            data_flat,
            partial_sums,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Stage 2: Final reduction (small, use PyTorch)
        output = partial_sums.sum().to(torch.float32)

        return output
