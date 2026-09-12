## 2026-03-31 - Vectorized Outpaint Padding Mask Computation
**Learning:** Per-pixel double loops in Python over image dimensions (`d2` height x `d3` width) incur massive interpreter overhead (over 2.8s for 1024x1024 images). Broadcasting 1D index tensors (`torch.arange`) with `torch.minimum` and `torch.clamp` computes spatial distances in C++/CUDA natively.
**Action:** Replace nested image pixel loops in node processing methods with PyTorch 1D index tensor broadcasting and elementwise tensor operations, keeping `device=image.device` for zero CPU/GPU overhead.
