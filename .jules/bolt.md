## 2026-06-26 - Optimized interpolation with torch.lerp
**Learning:** Manual linear interpolation using `a * (1 - w) + b * w` is significantly slower than `torch.lerp(a, b, w)` in PyTorch, especially on CPU where it can be ~70% slower for standard tensor shapes. `torch.lerp` is a fused operation that reduces overhead.
**Action:** Always prefer `torch.lerp(start, end, weight)` for linear interpolations between tensors to improve performance and numerical stability.
