## 2026-03-31 - Insecure Deserialization in PyTorch Checkpoint and Statistics Loading
**Vulnerability:** `CLIPEmbeddingNoiseAugmentation.__init__` in `comfy/ldm/modules/encoders/noise_aug_modules.py` loaded external statistics files using `torch.load(clip_stats_path)` without `weights_only=True`, allowing arbitrary Python code execution if loaded from an untrusted file.
**Learning:** Legacy `torch.load` defaults to standard Python `pickle` unpickling unless `weights_only=True` is explicitly passed, enabling malicious `__reduce__` payloads to execute shell commands or arbitrary code during deserialization.
**Prevention:** Always pass `weights_only=True` to `torch.load(...)` when deserializing model weights, embeddings, or numerical statistic files.
