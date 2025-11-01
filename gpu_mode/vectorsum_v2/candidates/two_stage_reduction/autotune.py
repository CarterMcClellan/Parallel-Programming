#!/usr/bin/env python3
"""
Auto-tune two_stage_reduction candidate: Test all BLOCK_SIZE values and show performance breakdown
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import triton
import triton.language as tl
from reference import generate_input
from utils import clear_l2_cache
from task import input_t, output_t
from utils import DeterministicContext


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

    tl.store(partial_sums_ptr + pid, block_sum)


def test_kernel(data_flat, block_size):
    """Run kernel with specific BLOCK_SIZE"""
    n_elements = data_flat.numel()
    n_blocks = triton.cdiv(n_elements, block_size)

    partial_sums = torch.empty(n_blocks, dtype=torch.float64, device=data_flat.device)
    grid = (n_blocks,)
    sum_reduction_kernel_stage1[grid](
        data_flat,
        partial_sums,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return partial_sums.sum().to(torch.float32)


def benchmark(block_size, test_sizes, num_runs=50, warmup=5):
    """Benchmark a BLOCK_SIZE across different input sizes"""
    results = []

    for test_size in test_sizes:
        data = generate_input(size=test_size, seed=42)
        data_flat = data[0].flatten()

        # Warmup
        for _ in range(warmup):
            _ = test_kernel(data_flat, block_size)
            torch.cuda.synchronize()

        # Benchmark
        durations = []
        for _ in range(num_runs):
            clear_l2_cache()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = test_kernel(data_flat, block_size)
            end_event.record()

            torch.cuda.synchronize()
            duration_ms = start_event.elapsed_time(end_event)
            durations.append(duration_ms)

        mean_ms = sum(durations) / len(durations)
        min_ms = min(durations)

        results.append({
            'size': test_size,
            'mean': mean_ms,
            'min': min_ms
        })

    avg_mean = sum(r['mean'] for r in results) / len(results)
    return {'block_size': block_size, 'results': results, 'avg_mean': avg_mean}


def main():
    print("=" * 80)
    print("AUTO-TUNING: two_stage_reduction - BLOCK_SIZE Performance Breakdown")
    print("=" * 80)

    test_sizes = [
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        2097152,
        4194304,
        8388608
    ]
    block_sizes = [64, 128, 256, 512, 1024, 2048]

    print(f"\nTest sizes: {[f'{s:,}' for s in test_sizes]}")
    print(f"Block sizes: {block_sizes}")
    print(f"Runs per config: 50\n")
    print("=" * 80)
    print()

    all_results = []

    for block_size in block_sizes:
        print(f"BLOCK_SIZE = {block_size}")
        result = benchmark(block_size, test_sizes)
        all_results.append(result)

        for r in result['results']:
            print(f"  Size {r['size']:>8,}: {r['mean']:>7.4f}ms (min: {r['min']:.4f}ms)")
        print(f"  Average: {result['avg_mean']:.4f}ms")
        print()

    # Summary
    best = min(all_results, key=lambda x: x['avg_mean'])

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'BLOCK_SIZE':<12} {'Avg (ms)':<12} {'Speedup':<10}")
    print("-" * 80)

    for result in sorted(all_results, key=lambda x: x['avg_mean']):
        speedup = best['avg_mean'] / result['avg_mean']
        marker = " <- BEST" if result['block_size'] == best['block_size'] else ""
        print(f"{result['block_size']:<12} {result['avg_mean']:<12.4f} {speedup:<10.2f}x{marker}")

    print()
    print("=" * 80)
    print(f"Optimal BLOCK_SIZE: {best['block_size']} ({best['avg_mean']:.4f}ms)")
    print("=" * 80)
    print()
    print(f"To apply this configuration, update BLOCK_SIZE in solution.py:")
    print(f"  BLOCK_SIZE = {best['block_size']}")


if __name__ == "__main__":
    main()
