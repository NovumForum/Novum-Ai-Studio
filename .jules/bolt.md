## 2025-05-15 - Optimized GPU Memory Metric Retrieval
**Learning:** `torch.cuda.memory_stats()` is expensive because it constructs a large dictionary of all CUDA metrics (~150µs). Using specialized functions like `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` is significantly faster (~10µs) and avoids GC pressure.
**Action:** Always prefer specialized memory functions over `memory_stats()` when only specific metrics (like allocated or reserved bytes) are needed.
