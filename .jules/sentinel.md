## 2026-08-15 - Path Traversal in Model Preview Route
**Vulnerability:** Unsanitized file paths and unvalidated array indexing in `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` allowed arbitrary directory traversal outside allowed model folders and unhandled exceptions.
**Learning:** `os.path.join` on user-supplied filenames can traverse directories if `..` sequences are included unless verified with `os.path.commonpath`. Integer path params must also be checked against list bounds before indexing.
**Prevention:** Always normalize absolute paths and assert `os.path.commonpath([base_dir, target_file]) == base_dir` when resolving user-controlled paths.
