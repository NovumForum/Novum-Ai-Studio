## 2026-09-16 - Vectorizing ImagePadForOutpaint pixel loops
**Learning:** Replacing per-pixel nested loops with 1D index tensor broadcasting (`torch.arange`), `torch.minimum`, and `torch.clamp` in PyTorch yields massive performance gains (~118x speedup on 1024x1024 images, from ~1.6s down to ~13.5ms). Always check tensor operations for elementwise or broadcasting replacements before keeping nested loops.
**Action:** Look for nested Python `for i in range(...)` / `for j in range(...)` operating over tensor dimensions and replace with 1D tensor broadcasting.
