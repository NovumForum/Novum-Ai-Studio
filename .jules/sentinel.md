## 2026-06-26 - [Path Traversal and Insecure Deserialization in Dataset Nodes]
**Vulnerability:** Nodes in `comfy_extras/nodes_dataset.py` were vulnerable to path traversal through user-provided folder names, and `LoadTrainingDataset` used insecure `torch.load` which could lead to RCE.
**Learning:** Experimental or "extra" nodes may lack the standard security guards found in the core API, especially when handling file paths or deserializing data. Prefix handling also needs explicit sanitization (e.g., `os.path.basename`) to prevent directory escape.
**Prevention:** Always use a `safe_join` utility that validates `os.path.commonpath` for user-supplied paths, and explicitly set `weights_only=True` in `torch.load` calls for untrusted data.
