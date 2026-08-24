## 2025-05-18 - Path Traversal Prevention in Model Preview Endpoint
**Vulnerability:** Unsanitized user path parameter in `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` allowed accessing arbitrary files outside the model directory via relative path sequences (`../`).
**Learning:** `os.path.join` alone does not restrict file access within a target directory when relative path components are supplied in URLs.
**Prevention:** Always convert both base folder and target file paths to absolute paths (`os.path.abspath`) and check `os.path.commonpath([folder, full_filename]) == folder` before accessing files.
