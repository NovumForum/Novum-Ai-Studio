## 2026-09-28 - Model Preview Path Traversal & Index Out-of-Bounds Safeguards
**Vulnerability:** The experimental `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` endpoint parsed `path_index` with `int(...)` without validating bounds against the folder paths list, and appended `filename` directly to `folder` without checking for path traversal (`../`).
**Learning:** Experimental API routes in `app/` often mirror production routes but may lack standard input validation or path boundary checks.
**Prevention:** Always validate URL path indices against array lengths and enforce `os.path.commonpath((abs_folder, full_filename)) == abs_folder` on file preview or download handlers.
