## 2025-06-28 - Path Traversal in Dataset Nodes
**Vulnerability:** Nodes in `comfy_extras/nodes_dataset.py` allowed users to specify folder names that could be used to access or save files outside the designated `input` and `output` directories via `../` sequences.
**Learning:** `os.path.join` does not prevent path traversal if the second argument is an absolute path or contains parent directory references. `os.path.commonpath` is needed to verify the resolved path remains within the base directory.
**Prevention:** Always use a `safe_join` utility that validates the resolved path against the base directory using `os.path.commonpath` and `os.path.abspath`.

## 2025-06-28 - Insecure torch.load calls
**Vulnerability:** Several `torch.load` calls lacked the `weights_only=True` parameter, making the application vulnerable to arbitrary code execution if a user provides a malicious pickle file.
**Learning:** Default `torch.load` behavior is insecure as it can execute arbitrary code during unpickling. This is particularly dangerous for nodes that load data from disk which might be user-controlled.
**Prevention:** Always set `weights_only=True` in `torch.load` unless there is a specific, well-justified reason to load non-tensor objects, and even then, only from trusted sources.
