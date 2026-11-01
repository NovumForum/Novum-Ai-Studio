# Sentinel's Journal

## 2025-05-18 - Path Traversal & Unvalidated Input in Model Preview Route
**Vulnerability:** The `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` route in `app/model_manager.py` directly cast `path_index` with `int(...)` without exception handling, accessed `folders[0][path_index]` without bounds checking, and joined `os.path.join(folder, filename)` without validating whether the resolved target resided within the model folder. This allowed path traversal outside the model directory as well as unhandled 500 crashes via invalid `path_index` values.
**Learning:** Route parameters matching greedy patterns like `{filename:.*}` or indexing into configuration lists (`path_index`) need strict sandboxing via `os.path.commonpath` and index bounds checking before any filesystem operations occur.
**Prevention:** Always validate integer route parameters with try-except (returning 400), verify index bounds against list length (returning 404), and enforce `os.path.commonpath([base_dir_abs, target_path_abs]) == base_dir_abs` (returning 403) whenever client-supplied file paths are resolved.
