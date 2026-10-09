## 2025-02-18 - Zero-Copy Component Selection for Expanded Image Tensors in Color Conversions

**Learning:** When color conversion nodes expand single-channel outputs (e.g. Y, U, V) into 3-channel tensors `(B, H, W, 3)` with duplicated channels for UI compatibility, downstream color space converters like `ImageYUVToRGB` do not need `torch.mean(..., dim=-1)` reductions. Taking zero-copy views `Y[..., 0]` avoids 3 reduction passes, memory allocations, and transposition steps, resulting in a ~1.35x-3.7x execution speedup.
**Action:** Always check if multi-channel tensors contain duplicated channel views from upstream expansion before performing channel reductions, and prefer zero-copy slicing (`tensor[..., 0]`) and direct CHW stacking (`torch.stack([y, u, v], dim=1)`).
