## 2025-03-06 - Vectorize FeatherMask operation

**Learning:** Sequential Python `for` loops in image/mask processing operations (e.g. iterating over individual pixels, rows, or columns) create massive interpreter overhead and slow down processing drastically on multi-channel/batch tensors.
**Action:** Replace iterative row/column slicing with 1D PyTorch tensor operations (`torch.linspace`, elementwise slice multiplication, or broadcasting) to execute full-tensor edge operations in a single vectorized step.
