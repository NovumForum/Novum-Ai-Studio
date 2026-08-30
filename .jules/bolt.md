## 2026-07-22 - Vectorize FeatherMask operation
**Learning:** Legacy Python loop indexing with negative indices (e.g., `-x`) in edge feathering loops maps `x=0` to index `0` and `x=1..right-1` to indices `-1..-(right-1)`. Vectorizing this with PyTorch tensor slicing requires separate treatment for index `0` and index range `(w-right+1):` with `.flip(0)` to preserve exact mathematical equivalence.
**Action:** When vectorizing Python loops with zero and negative slice indexing, carefully inspect indexing at `x=0` to ensure no off-by-one or slice boundary mismatch.
