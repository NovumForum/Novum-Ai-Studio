## 2025-02-17 - Secure Directory Scanning in Internal Routes
**Vulnerability:** Under certain configurations, unconfigured folder paths could return `None`. Calling `os.scandir(None)` in Python defaults to scanning the current working directory, leading to directory structure leakage of the application root.
**Learning:** This occurred because of implicit fallback behavior in Python's standard library where `None` maps to `.`. Unchecked inputs or unconfigured environment variables can bypass expected boundaries and fallback to the working directory.
**Prevention:** Always explicitly validate that any folder or path argument used in file operations (like `os.scandir`, `os.listdir`, `os.walk`) is not `None` and exists as a valid directory via `os.path.isdir()` before execution.
