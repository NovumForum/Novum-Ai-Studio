# Bolt's Performance Journal

⚡ Bolt's Daily Journal of Performance Bottlenecks and Learnings in ComfyUI.

## 2025-08-10 - Vectorizing PyTorch Batch Resizing
**Learning:** In ComfyUI, when adjusting tensor batch sizes during sampler or mask preprocessing, the original function `resize_to_batch_size` iteratively copied slices in a Python loop. Iterative sub-tensor indexing/assignment in PyTorch introduces significant Python object wrapping and C/C++ boundary crossings, causing huge latency overheads even for relatively small batch sizes. We can eliminate this bottleneck entirely by pre-generating index coordinates and performing a single vectorized index lookup (`tensor[indices]`), yielding over ~8x speedups.
**Action:** Always prefer single-step vectorized tensor lookups or gather operations over iterative PyTorch slicing and slice-assignment in Python loops.
