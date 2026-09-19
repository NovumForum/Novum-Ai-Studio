# Sentinel Security Journal

## 2026-09-19 - Path Traversal and Unhandled Exceptions in Model Preview Endpoint

**Vulnerability:** The experimental model preview endpoint (`GET /experiment/models/preview/{folder}/{path_index}/{filename:.*}`) parsed `path_index` directly with `int()` without bounds checking or exception handling, causing unhandled 500 errors on invalid inputs or out-of-bound indices. Furthermore, it joined `folder` and `filename` without verifying that the resolved path stayed within `folder`, allowing directory traversal to access arbitrary files on the system.

**Learning:** When endpoint handlers accept variable subpaths or index parameters from URL matches, direct type casting without `try/except` and unvalidated path concatenation (`os.path.join`) can lead to HTTP 500 crashes and path traversal security vulnerabilities.

**Prevention:** Always wrap integer route parameters in explicit validation logic (`try/except (ValueError, TypeError)`), check index range boundaries, and validate file paths using `os.path.commonpath([abs_root, abs_target]) == abs_root` before performing file operations.
