# Palette's Journal - Critical UX Learnings

## 2026-03-01 - Node Schema Metadata for Image Loaders
**Learning:** Core node schemas in ComfyUI power UI discovery and tooltips directly through `DESCRIPTION`, `OUTPUT_TOOLTIPS`, and `INPUT_TYPES` tooltip attributes parsed by the frontend API server (`server.py` `/object_info`). For multi-output nodes like `LoadImage` (`IMAGE`, `MASK`), output tooltips clarify that the mask represents the alpha channel (or empty mask when absent), eliminating confusion for users expecting mask outputs.
**Action:** When adding or updating node schemas, define `DESCRIPTION`, `OUTPUT_TOOLTIPS` matching `RETURN_TYPES` order/length, and input parameter tooltips to ensure clear UI tooltips and search indexing.
