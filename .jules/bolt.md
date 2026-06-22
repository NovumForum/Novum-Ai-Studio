## 2025-05-15 - [Optimize GPU memory metric retrieval]
**Learning:** Using `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` (and their equivalents for XPU, NPU, MLU) is significantly more efficient than calling `memory_stats()`. `memory_stats()` constructs a large dictionary containing dozens of metrics, which is unnecessary when only current allocation and reservation are needed.
**Action:** Always prefer direct memory accessor functions over `memory_stats()` in hot paths or frequent monitoring loops.
