## 2025-06-29 - Path Traversal and Insecure Deserialization in Dataset Nodes
**Vulnerability:** Experimental dataset nodes in `comfy_extras/nodes_dataset.py` allowed arbitrary path traversal via user-supplied folder names and used insecure `torch.load()` on training shards.
**Learning:** Experimental features often lack the standard security guards (like `safe_join`) found in the core server, making them high-value targets for security audits.
**Prevention:** Use a `safe_join` utility that validates the resolved absolute path against the expected base directory using `os.path.commonpath`. Always set `weights_only=True` in `torch.load()` when loading data that doesn't strictly require custom class restoration.
