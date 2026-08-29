
## 2025-05-18 - Vectorizing Batch Slicing and Concatenation in PyTorch Nodes
**Learning:** Python list comprehensions and iterative `torch.stack()` loops over batch dimensions in node executions introduce significant overhead compared to direct PyTorch tensor slicing (`image[..., :3]`) and batched `torch.cat()`. Vectorizing `SplitImageWithAlpha` and `JoinImageWithAlpha` sped up execution by up to ~4.6x.
**Action:** When working on PyTorch batch processing nodes, avoid Python `for` loops across the batch dimension (`for i in range(batch_size)`). Instead, slice batch tensors directly or use broadcasted batch operations.
