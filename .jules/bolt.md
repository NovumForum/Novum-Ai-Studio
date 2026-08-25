## 2025-02-25 - Caching Recursive Bayer Matrix Computation
**Learning:** In image post-processing workflows, helper routines like `normalized_bayer_matrix` can be called repeatedly per batch/frame inside dithered quantization loops, creating unnecessary `np.bmat` allocations and recursive overhead.
**Action:** Extract constant matrix generation functions to module-level scope and decorate with `@functools.lru_cache` to reuse computed matrices safely.
