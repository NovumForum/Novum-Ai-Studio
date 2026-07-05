## 2025-05-15 - [Path Traversal in Dataset Nodes]
**Vulnerability:** Nodes in `comfy_extras/nodes_dataset.py` using `io.String.Input` for folder paths were vulnerable to path traversal because they concatenated user strings directly with base directories without validation.
**Learning:** Unlike `io.Combo.Input` which restricts choices to a predefined list, `io.String.Input` allows arbitrary user-provided strings, making explicit server-side path validation (e.g., `safe_join`) mandatory.
**Prevention:** Always use a `safe_join` utility that verifies the resolved path remains within the intended base directory using `os.path.commonpath`.

## 2025-05-15 - [Insecure Deserialization in Training Shards]
**Vulnerability:** `LoadTrainingDataset` used `torch.load` on data shards without `weights_only=True`, potentially allowing arbitrary code execution if a user loaded a malicious dataset.
**Learning:** Even internal or experimental data formats like `.pkl` shards should be treated as untrusted inputs if they are loaded from user-specified directories.
**Prevention:** Default to `weights_only=True` for all `torch.load` calls unless complex Python objects are strictly required and the source is fully trusted.
