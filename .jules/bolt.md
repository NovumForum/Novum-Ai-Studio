## 2025-05-18 - PyTorch Tensor Vectorization in FeatherMask
**Learning:** Python per-pixel/per-slice loops over PyTorch tensors create significant interpreter overhead. Vectorizing tensor edge ramp multiplications using `torch.linspace` and 1D index tensor indexing eliminates loop overhead and provides ~5.5x-7.7x speedups.
**Action:** When working on tensor manipulation or image/mask processing nodes, look for Python `for` loops operating on tensor slice indices and replace them with vectorized 1D PyTorch ramps or index tensor slicing.
