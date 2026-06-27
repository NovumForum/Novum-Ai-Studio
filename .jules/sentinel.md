## 2025-05-15 - [Critical] Path Traversal and Insecure Deserialization in Dataset Nodes

**Vulnerability:**
1. Path Traversal: Multiple nodes in `comfy_extras/nodes_dataset.py` (e.g., `LoadImageDataSetFromFolderNode`, `SaveImageDataSetToFolderNode`) used `os.path.join` with user-supplied folder names without validation, allowing access to files outside `input` and `output` directories via `../` sequences.
2. Insecure Deserialization: `torch.load` was used on training dataset shards and CLIP stats without `weights_only=True`, which could lead to arbitrary code execution (RCE) if a malicious user provided a crafted `.pkl` or `.pt` file.

**Learning:**
Experimental and extra nodes (especially those handling datasets or manual file paths) often lack the rigorous path validation found in core nodes. `os.path.commonpath` is a reliable way to verify that a joined path stays within a target base directory, but it can raise `ValueError` on Windows if paths are on different drives.

**Prevention:**
1. Always use a `safe_join` utility that validates the resulting absolute path against the base directory using `os.path.commonpath`. Include error handling for `ValueError` to prevent crashes in cross-drive scenarios on Windows.
2. Explicitly set `weights_only=True` in all `torch.load` calls unless there is a specific, documented reason to allow arbitrary object deserialization.
