## 2025-05-15 - Robust Path Traversal Protection with safe_join
**Vulnerability:** User-controlled folder names and file prefixes in dataset nodes were used directly with `os.path.join`, allowing path traversal to read or write files outside intended input/output directories.
**Learning:** Standard `os.path.join` does not prevent traversal if an argument is an absolute path or contains `..` segments. Combining `os.path.abspath` with `os.path.commonpath` provides a robust check that respects path boundaries.
**Prevention:** Always use a utility like `safe_join` when combining a trusted base directory with untrusted user input. Ensure both paths are absolute before comparison to avoid bypasses.
