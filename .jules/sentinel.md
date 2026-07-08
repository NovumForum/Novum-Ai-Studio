## 2025-05-15 - Path Traversal in Experimental Dataset Nodes
**Vulnerability:** Several experimental nodes in `comfy_extras/nodes_dataset.py` used `os.path.join` with user-provided strings for folder names, allowing potential path traversal to access or write files outside the intended input/output directories.
**Learning:** V3 API nodes using `io.String.Input` for paths require explicit validation because unlike `io.Combo.Input`, they allow arbitrary string input. A centralized `safe_join` utility is necessary to consistently enforce directory boundaries.
**Prevention:** Always use a `safe_join` utility that validates the final resolved path against the base directory using `os.path.commonpath` after `os.path.abspath`.

## 2025-05-15 - Insecure Deserialization in Training Shards
**Vulnerability:** The `LoadTrainingDataset` node used `torch.load` without `weights_only=True` to load `.pkl` shards, which could lead to arbitrary code execution if a user is tricked into loading a malicious dataset.
**Learning:** Even internal data formats like training shards should be loaded securely to prevent RCE, especially when the file paths can be influenced by user input.
**Prevention:** Always set `weights_only=True` in `torch.load` unless loading complex custom classes is strictly required and the source is trusted.
