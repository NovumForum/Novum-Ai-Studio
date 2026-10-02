## 2025-05-20 - Vectorizing PyTorch Tensor Edge Modifications

**Learning:** Slicing 2D/3D/4D PyTorch tensors with 1D PyTorch ramps (`torch.arange` broadcasted or sliced) replaces Python index `for` loops, eliminating Python interpreter overhead and yielding ~3.3x speedups on CPU.
**Action:** Always check `for x in range(...)` loops that multiply or slice tensors element-by-element in custom PyTorch node implementations, and replace them with vectorized `torch.arange` ramp broadcasting.
