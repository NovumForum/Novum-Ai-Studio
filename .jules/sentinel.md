# Sentinel's Journal 🛡️

## 2025-05-15 - Path Traversal in Dataset Nodes
**Vulnerability:** Several nodes in `comfy_extras/nodes_dataset.py` accepted a `folder` or `folder_name` string that was directly joined with base input/output directories. An attacker could use `../../` or absolute paths to access arbitrary directories on the host system.
**Learning:** Using `os.path.join` with an absolute path as the second argument discards the first argument, and `../` sequences are not automatically resolved. `os.path.commonpath` is an effective way to verify that a resolved path remains within a boundary, but both the base and the target path MUST be normalized using `os.path.abspath` beforehand to avoid false positives (e.g. from trailing slashes).
**Prevention:** Always normalize both base and target paths using `os.path.abspath()` and verify the boundary using `os.path.commonpath()` before performing filesystem operations on user-provided path components.
