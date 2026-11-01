# Palette's Journal - Critical UX Learnings

## 2026-03-01 - Schema Discoverability and Tooltip Metadata in Core Nodes
**Learning:** In ComfyUI's Python-defined node schemas, core UI searchability and contextual guidance rely directly on class-level attributes (`DESCRIPTION`, `SEARCH_ALIASES`, `OUTPUT_TOOLTIPS`) and input configuration dictionaries containing `tooltip`. Missing metadata on foundational loader nodes like `VAELoader` impairs new user onboarding and node palette search discoverability. Testing schema metadata via AST parsing enables resilient testing without requiring heavyweight runtime dependencies like PyTorch or CUDA.
**Action:** Always supply descriptive `SEARCH_ALIASES`, `DESCRIPTION`, `OUTPUT_TOOLTIPS`, and parameter `tooltip` when touching or standardizing node schemas, and test them with AST assertions in `tests-unit/`.
