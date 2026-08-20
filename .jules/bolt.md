## 2026-03-30 - Vectorized Mask Growth with PyTorch Max Pooling

**Learning:** Replacing SciPy's CPU-bound `grey_dilation`/`grey_erosion` loops with batched PyTorch `max_pool2d` (or `-max_pool2d(-mask)`) with replicate padding eliminates PyTorch-to-NumPy array conversions and sequential CPU loops, keeping execution native on the target tensor device (CPU/GPU) with 100% exact mathematical equivalence (`0.0` max difference).

**Action:** For morphological operations or mask operations in PyTorch, use native batched tensor pooling (`max_pool2d`) instead of delegating to SciPy or NumPy in loops.
