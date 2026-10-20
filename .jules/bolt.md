# Bolt's Journal - Critical Learnings

## 2025-02-20 - PyTorch Batch Unrolling Pattern
**Learning:** Node implementations in ComfyUI that extract alpha or slice image channels often use Python list comprehensions over batch dimensions followed by `torch.stack`. Operating directly on 4D PyTorch tensors (`[B, H, W, C]`) using multi-dimensional slicing (`image[..., :3]`, `image[..., 3]`) avoids Python loop unrolling and unnecessary `torch.stack` memory re-allocations, providing ~3x speedup.
**Action:** When inspecting compositing or image manipulation nodes, check if operations can be expressed directly as 4D PyTorch tensor slices across batch dimensions.
