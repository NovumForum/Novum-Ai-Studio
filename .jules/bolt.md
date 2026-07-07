## 2025-05-15 - Optimization of Linear Interpolation (lerp)

**Learning:** Manual linear interpolation using the formula `a * (1 - w) + b * w` or `a + (b - a) * w` is common in sampling and EMA loops but is less efficient than the native `torch.lerp(a, b, w)`. PyTorch's `lerp` uses fused kernels which reduce memory bandwidth and improve speed, especially on GPUs. Additionally, short-circuiting EMA updates when `beta == 1.0` avoid redundant tensor operations and allocations.

**Action:** Always prefer `torch.lerp` for linear interpolation between tensors. Check for identity weights (e.g., `beta=1.0` or `strength=0.0`) to skip computations entirely in hot paths.
