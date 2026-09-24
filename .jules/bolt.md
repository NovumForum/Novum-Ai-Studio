## Performance Learnings

### 2025-05-24 - Vectorizing Batch Operations in Image/Alpha Compositing
**Learning:** In PyTorch node implementations, iterating over batch dimensions using Python list comprehensions (e.g. `[i[:,:,:3] for i in image]` or `[torch.cat(...) for i in range(batch_size)]`) followed by `torch.stack()` incurs heavy Python loop overhead and unneeded allocation of intermediate tensors. Replacing per-item Python loops with batched tensor slicing (`image[:, :, :, :3]`) and batched concatenation (`torch.cat((img, alpha_resized.unsqueeze(-1)), dim=-1)`) yields a ~5x speedup for `SplitImageWithAlpha` and ~1.5x speedup for `JoinImageWithAlpha`.
**Action:** When working on batch image or mask nodes, always check if loop iterations over the batch dimension `[B, H, W, C]` can be replaced with vectorized tensor slicing and operations across the whole batch tensor at once.
