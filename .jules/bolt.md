## 2026-06-23 - [Efficient GPU Memory Metric Retrieval]
**Learning:** Using `torch.cuda.memory_stats()` is inefficient when only specific metrics like allocated or reserved memory are needed. `memory_stats()` constructs a large Python dictionary containing dozens of statistics, incurring unnecessary CPU overhead and memory allocations.
**Action:** Always prefer specialized functions like `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` for fetching specific GPU memory metrics.
