## 2025-05-20 - Input/Output/Temp Directory Existence Check Before Directory Scan
**Vulnerability:** Calling `os.scandir` on missing/unconfigured internal file directories (`/internal/files/{directory_type}`) caused uncaught server exceptions (`FileNotFoundError` or `NotADirectoryError`), resulting in 500 Internal Server Errors and exposing internal error traces.
**Learning:** `get_directory_by_type` may return `None` or a path that does not exist on disk if user configurations or environment paths are missing.
**Prevention:** Always validate directory existence and type with `os.path.isdir(directory)` before calling filesystem scanning functions like `os.scandir` or `os.listdir`, returning a standard 404 response if the directory is missing.
