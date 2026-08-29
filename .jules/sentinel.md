## 2026-03-23 - Insecure PyTorch Checkpoint Deserialization in CLIP Noise Augmentation
**Vulnerability:** `CLIPEmbeddingNoiseAugmentation.__init__` loaded statistics via `torch.load` without `weights_only=True`, exposing the application to Remote Code Execution (RCE) if `clip_stats_path` was specified pointing to an untrusted pickle payload.
**Learning:** PyTorch `torch.load` defaults to arbitrary unpickling unless `weights_only=True` is explicitly passed.
**Prevention:** Always pass `weights_only=True` to `torch.load` calls across model loaders and encoders to ensure only tensor data and non-code primitives are deserialized.
