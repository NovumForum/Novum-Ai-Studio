## 2026-09-11 - Path Traversal in Wildcard Route Handlers
**Vulnerability:** Path traversal in `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` allowed accessing files outside model directories and unhandled non-integer/out-of-bounds `path_index` inputs caused HTTP 500 errors.
**Learning:** Client-side HTTP clients normalize `../` path segments in URLs, which can cause traversal paths to match different route templates before reaching the server unless URL-encoded (`%2f`).
**Prevention:** Always validate `path_index` types and bounds explicitly, and resolve `os.path.abspath` to verify `os.path.commonpath([abs_folder, full_filename]) == abs_folder`.
