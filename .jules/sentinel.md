## 2025-02-18 - Enforce weights_only=True in PyTorch Deserialization
**Vulnerability:** `CLIPEmbeddingNoiseAugmentation.__init__` used `torch.load(clip_stats_path)` without `weights_only=True`, exposing the application to arbitrary code execution if an untrusted or malicious `.pt`/`.pkl` clip stats file was supplied.
**Learning:** Legacy PyTorch `torch.load` calls without `weights_only=True` fall back to standard Python pickle deserialization, allowing arbitrary object construction and command execution via `__reduce__` methods.
**Prevention:** Always pass `weights_only=True` to `torch.load` when loading tensor checkpoints or state dictionaries to enforce safe primitive/tensor unpickling.
