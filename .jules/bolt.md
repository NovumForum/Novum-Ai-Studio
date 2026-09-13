## 2025-02-18 - Vectorizing Python Negative-Index Edge Loops in PyTorch Tensors
**Learning:** When vectorizing Python loops using negative indices like `output[:, :, -x]` over `range(N)`, Python treats `x=0` as index `0` (`-0 == 0`) rather than the end of the tensor. Vectorization must split index `0` from negative index slices (`-indices` for `indices = 1..N-1`) to avoid incorrectly overwriting column/row 0 multiple times.
**Action:** Always handle `x=0` index specially when converting legacy negative-indexed loop operations to PyTorch tensor slicing.
