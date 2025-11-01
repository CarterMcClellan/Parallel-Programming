"""
Submission entry point

This imports the best candidate solution for evaluation.
To change which candidate is used, modify the import below.
"""

# Import the best candidate solution
# Options: atomic_add, two_stage_reduction
from candidates.two_stage_reduction.solution import custom_kernel

__all__ = ['custom_kernel']