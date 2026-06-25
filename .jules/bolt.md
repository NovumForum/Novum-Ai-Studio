# Bolt's Journal - Performance Optimizations

## 2025-05-15 - Linear Interpolation with torch.lerp
**Learning:** Using `torch.lerp(input, end, weight)` is preferred over manual `(1 - w) * a + w * b` as it is more numerically stable and potentially faster due to specialized kernel implementation in PyTorch.
**Action:** Replace manual linear interpolation in `ConditioningAverage` and `LatentBlend` with `torch.lerp`.
