# Palette's UX Journal

## 2025-02-18 - Schema Parameter Descriptions & Help Discovery
**Learning:** ComfyUI modern V3 schemas leverage tooltips, search aliases, and node-level descriptions to render user-friendly guidance in-context directly in the UI. Without these, users face layout/operation ambiguity, especially on scaling or post-processing nodes where aspect-ratio logic is crucial.
**Action:** Always enrich node schemas with `display_name`, `search_aliases`, descriptive parameter `tooltip`s, and clear `description` metadata to aid discoverability and accessibility.
