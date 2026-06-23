## 2025-05-15 - Hardening Dataset Nodes against RCE and Path Traversal
**Vulnerability:** Dataset nodes were vulnerable to Remote Code Execution (RCE) via insecure `torch.load` calls on untrusted training shards and path traversal via unvalidated folder/prefix inputs.
**Learning:** Experimental or utility nodes often lack the rigorous security validation found in core components. `torch.load` defaults to using pickle, which is inherently unsafe unless `weights_only=True` is used. Path traversal remains a high-impact risk when user strings are joined to system paths.
**Prevention:** Always use `weights_only=True` for PyTorch loading of non-model data (and models when possible). Use a robust `safe_join` pattern with `os.path.commonpath` to validate all user-supplied file/folder paths.
