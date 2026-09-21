## 2026-03-30 - Vectorizing Image Outpainting Feathering Mask

**Learning:** In `nodes.py`, `ImagePadForOutpaint.expand_image` computed per-pixel feathering values using $O(H \times W)$ nested Python loops (`for i in range(d2): for j in range(d3)`), causing a ~1.6 second latency penalty per 1024x1024 image execution. Replacing nested loops with 1D index tensor broadcasting (`torch.arange`) and elementwise tensor ops (`torch.minimum`, `torch.clamp`) reduces execution time to ~3.3ms (~480x speedup).

**Action:** Whenever generating 2D distance or feathering fields in image/mask nodes, construct 1D coordinate tensors (`torch.arange`) along height and width dimensions and rely on PyTorch tensor broadcasting instead of Python `for` loops.
