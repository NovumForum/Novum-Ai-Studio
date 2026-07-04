## 2025-05-15 - [Fused Kernel Optimization]
**Learning:** Replacing manual linear interpolation arithmetic (`a * w + b * (1 - w)`) with PyTorch's fused `torch.lerp(b, a, w)` provides a significant performance boost (measured at ~7.75x speedup for large tensors on CPU) by reducing intermediate memory allocations and leveraging kernel fusion.
**Action:** Always prefer `torch.lerp` for weighted sums and EMA updates.

## 2025-05-15 - [Short-Circuit Identity Transforms]
**Learning:** In weight adapters (OFT/BOFT), the multiplier or strength is often set to 1.0 (identity). Manual interpolation still allocates identity matrices and performs tensor multiplications in this case.
**Action:** Add explicit `if multiplier != 1.0` checks to skip identity matrix generation and interpolation arithmetic when the transform is a no-op.
