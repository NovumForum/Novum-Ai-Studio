## 2025-05-15 - Torch.lerp for Performance and Stability
**Learning:** Manual linear interpolation using `a * alpha + b * (1.0 - alpha)` is a common pattern in diffusion models but is less efficient than `torch.lerp(b, a, alpha)`. `torch.lerp` is often implemented as a fused kernel, reducing intermediate memory allocations and improving numerical stability.
**Action:** Use `torch.lerp(start, end, weight)` for all linear interpolation tasks on tensors, especially in hot paths like CFG calculation and sampling loops.
