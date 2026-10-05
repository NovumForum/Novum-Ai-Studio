## 2026-03-31 - Path Traversal via Relative Path Resolution
**Vulnerability:** `folder_paths.get_full_path` attempted path normalization via `os.path.relpath(os.path.join("/", filename), "/")`, which still allowed escaping folder boundaries when joined with target model directories.
**Learning:** `os.path.relpath(os.path.join("/", filename), "/")` does not enforce strict root containment when concatenated with arbitrary base folders.
**Prevention:** Always enforce `os.path.commonpath((full_path, abs_folder)) == abs_folder` on all file path resolution utilities before checking file existence or returning paths.
