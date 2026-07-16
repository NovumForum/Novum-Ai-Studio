## 2025-05-15 - Fused Linear Interpolation with torch.lerp
**Learning:** Replacing manual linear interpolation patterns (`a + (b - a) * w` or `a * (1 - w) + b * w`) with `torch.lerp` provides a measurable speedup (up to 2.8x in sampler hot paths) by leveraging fused kernels and reducing intermediate tensor allocations.
**Action:** Always prefer `torch.lerp` for linear interpolation between tensors in performance-critical paths.
