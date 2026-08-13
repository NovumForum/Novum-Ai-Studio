## 2026-07-22 - Secure directory scanning on the internal /files endpoint
**Vulnerability:** In Python, calling `os.scandir(None)` falls back to scanning the current working directory (`'.'`), which can leak root/app folders if a directory configuration utility returns `None`.
**Learning:** The `/internal/files/{directory_type}` endpoint returned `os.scandir(directory)` directly on the output of `get_directory_by_type`. If the requested directory type was valid but unconfigured (returning `None`), this scanned the root execution directory, leaking files and paths.
**Prevention:** Always validate resolved directory paths with `not None` and `os.path.isdir()` before invoking scanning or traversal APIs like `os.scandir` or `os.walk`.
