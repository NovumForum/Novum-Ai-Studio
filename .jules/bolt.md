## 2025-05-18 - Vectorized Alpha Splitting and Joining in Compositing
**Learning:** PyTorch nodes that iterate across batch dimensions using Python list comprehensions and `torch.stack` introduce significant interpreter overhead and unnecessary tensor re-allocations.
**Action:** Always prefer direct multidimensional slicing (e.g. `image[..., :3]`) and batched concatenation across batch dimensions (`torch.cat((img, alpha.unsqueeze(-1)), dim=-1)`) over `for` loops and `torch.stack`.
