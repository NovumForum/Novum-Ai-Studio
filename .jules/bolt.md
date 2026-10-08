## 2026-03-06 - Vectorize PyTorch Tensor Loop Slicing with 1D Ramps
**Learning:** Iterating over PyTorch tensor slices using Python `for` loops in nodes like `FeatherMask` causes excessive Python-to-C++/CUDA dispatch overhead (e.g. ~1000 kernel launches per call for a 256px border).
**Action:** Replace per-index Python loop slicing with vectorized 1D tensor ramp operations (`torch.arange`) broadcasted across dimensions to achieve ~30x speedups.
