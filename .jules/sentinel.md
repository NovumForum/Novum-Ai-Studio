## 2025-05-15 - Robust Path Traversal Protection
**Vulnerability:** User-controlled folder names were used in `os.path.join` without validation, allowing path traversal (e.g., `../../etc/passwd`).
**Learning:** Centralizing path validation in a `safe_join` utility using `os.path.commonpath` is more robust than string-based prefix checks, as it handles symlinks and various path formats correctly when combined with `os.path.abspath`.
**Prevention:** Always use a safe joining utility for any path construction involving user input, ensuring the final resolved path stays within the intended base directory.
