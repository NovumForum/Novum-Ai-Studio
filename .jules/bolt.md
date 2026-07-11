## 2025-05-15 - Fused Linear Interpolation with torch.lerp
**Learning:** Replaced manual linear interpolation (e.g., `x * alpha + y * (1 - alpha)`) with `torch.lerp(y, x, alpha)` across critical paths (EMA, samplers, adapters). `torch.lerp` is a fused kernel that is measurably faster and reduces intermediate tensor allocations.
**Action:** Always prefer `torch.lerp` for linear interpolation in PyTorch codebases.

## 2025-05-15 - Short-circuiting EMA and Adapter computations
**Learning:** Added `if multiplier != 1.0:` short-circuits to skip `torch.eye` allocation and interpolation logic when the strength is at its maximum (identity transformation).
**Action:** Identify and short-circuit hot paths where a parameter value (like 1.0 or 0.0) makes the operation a no-op.
