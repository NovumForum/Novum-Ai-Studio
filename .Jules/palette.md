## 2025-05-20 - ResolutionSelector Search Discovery & Node Tooltips
**Learning:** ComfyUI nodes like `ResolutionSelector` often lack `search_aliases`, making them hard to discover when users search for intuitive terms like "aspect ratio", "megapixels", "dimensions", or "image size".
**Action:** Always add `search_aliases` and check input/output parameter tooltips when updating node schema definitions in `comfy_extras`.
