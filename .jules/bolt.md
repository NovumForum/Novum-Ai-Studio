## 2025-05-15 - Optimize CUDA memory queries
**Learning:** `torch.cuda.memory_stats()` returns a large dictionary containing extensive diagnostic information, which is expensive to construct. For common memory monitoring (allocated and reserved bytes), `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` are much more efficient as they return single values directly.
**Action:** Use targeted memory query functions instead of parsing `memory_stats()` when only specific current values are needed.
