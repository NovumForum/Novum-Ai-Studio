# Palette's Journal

## 2025-05-18 - Node Schema Metadata & Tooltips Testing Pattern
**Learning:** ComfyUI core nodes in `nodes.py` directly feed the web UI canvas and node search with metadata (`DESCRIPTION`, `SEARCH_ALIASES`, `OUTPUT_TOOLTIPS`, `RETURN_NAMES`, and input parameter `tooltip`). Since unit test environments may not have heavy runtime dependencies like `torch` installed, schema metadata can be reliably tested across environments using Python's built-in `ast` module without requiring model dependencies or initialization.
**Action:** Always enrich node schemas with concise, context-aware input/output tooltips and search aliases, and verify them via AST parsing unit tests under `tests-unit/`.
