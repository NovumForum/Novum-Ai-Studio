## 2025-05-14 - Optimized Linear Interpolation with torch.lerp
**Learning:** Manual linear interpolation `(1 - w) * a + w * b` or `a + (b - a) * w` in PyTorch can be significantly slower than `torch.lerp(a, b, w)`. `torch.lerp` is a fused kernel that reduces memory bandwidth and improves numerical stability, especially for large tensors or high-frequency operations like CFG and sampling loops.
**Action:** Always prefer `torch.lerp(start, end, weight)` for linear interpolation in hot paths. Ensure `start` and `end` are on the same device and `weight` is a scalar or matching tensor.
