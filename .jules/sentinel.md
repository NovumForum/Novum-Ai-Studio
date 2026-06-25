## 2025-05-15 - Path Traversal and Insecure Deserialization in Dataset Nodes
**Vulnerability:** Multiple nodes in `comfy_extras/nodes_dataset.py` were vulnerable to path traversal via user-controlled folder names and insecure deserialization through `torch.load`.
**Learning:** Dataset management nodes often handle file system paths and serialized data, making them high-value targets for path traversal and RCE. Relying on simple `os.path.join` with user input is insufficient.
**Prevention:** Always use a `safe_join` pattern with `os.path.commonpath` to enforce boundary checks for user-provided paths. Explicitly set `weights_only=True` in `torch.load` to mitigate pickle-based RCE.
