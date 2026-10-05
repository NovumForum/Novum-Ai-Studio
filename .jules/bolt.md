# Bolt's Journal - Critical Learnings

## 2025-05-18 - Tile List Merging Accumulator Allocation
**Learning:** `ImageMergeTileList` accumulated tile weights into a 4D tensor with shape `(b, h, w, 1)` matching the batch size `b`. Because tile weight masks are identical across all batch items, allocating `weights` with shape `(1, h, w, 1)` avoids redundant memory allocations and broadcasting overhead during canvas reconstruction, yielding ~1.17x speedup and reducing `weights` memory usage by $b \times$.
**Action:** When accumulating spatially invariant per-pixel weight masks across batch dimensions in image processing operations, initialize weight accumulators with a single batch dimension `(1, H, W, C)` and rely on PyTorch tensor broadcasting for final division.
