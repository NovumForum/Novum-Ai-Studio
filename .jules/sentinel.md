## 2026-09-21 - Path Traversal in Model Preview Route
**Vulnerability:** Unsanitized path parameters in `GET /experiment/models/preview/{folder}/{path_index}/{filename:.*}` allowed directory traversal outside of configured model directories.
**Learning:** `os.path.join(folder, filename)` does not prevent path traversal if `filename` contains relative parent components like `..`. Additionally, string to int conversions on `path_index` without error handling led to unhandled 500 server errors.
**Prevention:** Always resolve absolute paths using `os.path.abspath` and verify `os.path.commonpath([base_dir, target_path]) == base_dir` before accessing files. Safely handle type conversions and check array bounds on route parameters.
