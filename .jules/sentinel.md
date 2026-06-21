## 2025-05-22 - Path Traversal and Insecure Deserialization in Dataset Nodes
**Vulnerability:** Several dataset-related nodes in `comfy_extras/nodes_dataset.py` were vulnerable to path traversal through user-supplied folder names and filename prefixes. Additionally, `LoadTrainingDataset` was vulnerable to insecure deserialization via `torch.load`.
**Learning:** Experimental or "extra" nodes may not receive the same level of security scrutiny as core nodes, making them a common place for vulnerabilities like path traversal to persist.
**Prevention:** Always use a helper like `ensure_path_within` when joining user input with base directories. Explicitly set `weights_only=True` in `torch.load` to prevent arbitrary code execution, regardless of the default in the environment.
