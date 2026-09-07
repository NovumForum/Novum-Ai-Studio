## 2026-03-30 - Iterative PyTorch Block Construction for Bayer Dithering
**Learning:** Replacing recursive NumPy matrix building (`np.bmat`) with iterative PyTorch block concatenation (`torch.cat`) and avoiding unnecessary NumPy array memory copies accelerates Bayer dithering matrix generation by ~2.8x.
**Action:** When generating patterned or recursive matrices in PyTorch/PIL image processing nodes, construct matrices directly using PyTorch tensor operations (`torch.cat`, `torch.kron`) rather than NumPy array wrappers.
