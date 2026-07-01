## 2026-07-01 - [Fused Kernel Optimization in CFG Hot Path]
**Learning:** Classifier-Free Guidance (CFG) calculations are a significant hot path in diffusion models, executed multiple times per step. Manual interpolation `a + (b - a) * w` triggers multiple kernel launches and memory passes. Replacing this with `torch.lerp(a, b, w)` leverages fused kernels, reducing memory bandwidth pressure and improving numerical stability.
**Action:** Always prefer `torch.lerp` for linear interpolation in performance-critical paths, especially when working with large tensors (latents/conditioning).
