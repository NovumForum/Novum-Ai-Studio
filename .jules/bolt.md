## 2025-03-08 - Vectorizing Image Mask Dilation/Erosion in PyTorch

**Learning:** `GrowMask` node performed mask dilation/erosion using SciPy `grey_dilation` and `grey_erosion` inside per-mask Python loops, converting PyTorch tensors to NumPy arrays (`m.numpy()`). Re-implementing morphological operations directly in PyTorch using `F.pad`, `F.max_pool2d`, and tensor `torch.max`/`torch.min` vectorizes across batch dimensions and avoids CPU-GPU tensor transfers.
**Action:** Replace CPU SciPy loop operations on PyTorch mask/image tensors with native PyTorch tensor ops (`max_pool2d`, `min`, `max`, `pad`) to keep operations hardware-accelerated and batch-vectorized.
