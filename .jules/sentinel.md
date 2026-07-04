# Sentinel Journal

## 2025-05-15 - Path Traversal and Insecure Deserialization in Dataset Nodes
**Vulnerability:** User-controlled strings were used to construct file paths without validation in `comfy_extras/nodes_dataset.py`, allowing path traversal. Additionally, `torch.load` was used without `weights_only=True`, allowing potential arbitrary code execution.
**Learning:** Experimental nodes using `io.String.Input` for directory or file names are particularly susceptible to path traversal because they don't use the pre-filtered options provided by `io.Combo.Input`. Hardening these requires explicit validation using a `safe_join` pattern.
**Prevention:** Always use a `safe_join` utility that validates the resolved path against a base directory using `os.path.commonpath`. Always enable `weights_only=True` when using `torch.load` on data that could be user-provided or modified.
