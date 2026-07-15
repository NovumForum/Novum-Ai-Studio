## 2025-05-15 - Linear Interpolation Optimization with torch.lerp
**Learning:** Manual implementation of linear interpolation (e.g., `a + (b - a) * w` or `a * (1 - w) + b * w`) in PyTorch creates multiple intermediate tensors and requires multiple kernel launches. `torch.lerp` provides a fused operation that is more memory-efficient and faster.
**Action:** Always prefer `torch.lerp(input, end, weight)` for linear interpolations in hot paths like sampling and weight adapters.
