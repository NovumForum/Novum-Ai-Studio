## 2025-05-15 - Path Traversal and Insecure Deserialization in Dataset Nodes
**Vulnerability:** User-supplied folder names and prefixes in dataset nodes were used in `os.path.join` without validation, allowing path traversal. Additionally, `LoadTrainingDataset` used `torch.load` without `weights_only=True`, allowing arbitrary code execution via malicious pickle data.
**Learning:** Experimental or "extra" nodes often lack the rigorous security checks applied to the core engine. Path traversal is especially dangerous in file-handling nodes that interact with `input/` and `output/` directories.
**Prevention:** Use a `safe_join` utility that validates the resolved path using `os.path.commonpath`. Always set `weights_only=True` for `torch.load` when handling user-accessible data.
