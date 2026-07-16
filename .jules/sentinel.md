## 2026-07-16 - Improper directory creation in safe_join utility
**Vulnerability:** Path traversal via unsanitized user-controlled folder names in dataset nodes.
**Learning:** When implementing path traversal protection, using a helper like `safe_join` with an auto-mkdir flag (`create_dir=True`) on a path that includes a filename causes the OS to create a directory where the file was intended to be, breaking subsequent write operations with `IsADirectoryError`.
**Prevention:** Explicitly distinguish between directory joins (where auto-creation is safe) and file joins. For files, call `os.makedirs` on the parent directory first, then join the filename without the creation flag.
