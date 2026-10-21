## 2026-10-21 - Model Preview Path Traversal and Unhandled Exception Guard
**Vulnerability:** The `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` endpoint in `app/model_manager.py` did not validate that `path_index` was a valid integer or within bounds of `folders[0]`, nor did it check if `filename` escaped `folder` via path traversal (`..`).
**Learning:** Endpoints taking path dynamic parts and indices without strict validation can result in unhandled `ValueError` / `IndexError` (500 server errors) or directory traversal file leakage.
**Prevention:** Always validate integer conversion and array index bounds before indexing lists, and use `os.path.abspath` + `os.path.commonpath` to enforce directory containment before opening file paths constructed from user parameters.
