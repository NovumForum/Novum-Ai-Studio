## 2026-03-01 - Vectorizing FeatherMask in nodes_mask.py
**Learning:** Python per-index loops over PyTorch tensor slices incur heavy per-iteration Python interpreter overhead and repeated sub-tensor slice creation. Vectorizing 1D gradient ramp operations using `torch.arange` and tensor slicing achieves ~3x-4x speedup while preserving exact numerical equivalence.
**Action:** When working on mask or image tensor gradient / ramp operations, replace Python scalar loops over dimensions with 1D `torch.arange` operations broadcasted or indexed directly onto the target tensor.
