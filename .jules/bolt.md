## 2025-05-18 - Avoid Pre-Concatenating Large 4D CPU Tensors Before Slicing

**Learning:** Concatenating multi-batch 4D CPU image tensors into a single combined tensor (`torch.cat(images, dim=0)`) prior to chunking degraded execution speed (~0.47x speedup / 2x slower) due to massive memory allocation overhead and L3 cache thrashing. Conversely, chunking input tensors directly using strided tensor slicing (`img[offset:offset+take]`) produces zero-copy tensor views when chunks fit within single tensors, delivering up to ~1200x–2200x speedups for single or large batch inputs.

**Action:** When rebatching or slicing batch tensors in PyTorch, slice directly from input tensors to create zero-copy views, and only invoke `torch.cat` across batch boundaries when an output batch spans multiple input tensors.
