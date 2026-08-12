# Sentinel's Journal - Critical Learnings Only

## 2026-08-12 - Path Traversal & Unhandled Exceptions in Experimental Routes
**Vulnerability:** Path traversal (arbitrary file preview) and server-crashing `IndexError` / `ValueError` exceptions on the `/experiment/models/preview/{folder}/{path_index}/{filename:.*}` endpoint.
**Learning:** Experimental endpoints can easily bypass standard input validation practices, leaving parameters like unconstrained file paths and array indexes unchecked. An index out of bounds crashed the service, and a traversable filepath could allow arbitrary system file inspection.
**Prevention:** Always enforce containment checks using `os.path.commonpath` against the resolved absolute base directory, and explicitly validate that array indices are integers within bounds before subscripting arrays.
