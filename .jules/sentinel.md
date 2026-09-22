## 2026-09-22 - Path Traversal in Model Preview Route
**Vulnerability:** The `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` endpoint accepted arbitrary path segments in `{filename:.*}` without validating that the target path remained inside the model folder.
**Learning:** Raw wildcards in route parameters (`{filename:.*}`) allow directory traversal sequences (`..`) to escape expected base directories unless explicitly restricted with `os.path.commonpath` or `os.path.abspath` verification.
**Prevention:** Always convert both base directory and requested file path to absolute paths and verify `os.path.commonpath([abs_base, abs_file]) == abs_base` prior to accessing files.
