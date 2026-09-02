## 2025-05-18 - Enforce weights_only=True in Dataset Loaders
**Vulnerability:** `LoadTrainingDataset.execute` in `comfy_extras/nodes_dataset.py` used unsafe `torch.load(f)` without `weights_only=True` to load dataset shard files (`.pkl`), allowing arbitrary code execution via crafted pickle payloads.
**Learning:** Legacy dataset loading routines often relied on default `torch.load` behavior without explicit `weights_only=True` restrictor flag.
**Prevention:** Always pass `weights_only=True` to `torch.load` when loading model weights, latent data, or dataset checkpoints from disk.
