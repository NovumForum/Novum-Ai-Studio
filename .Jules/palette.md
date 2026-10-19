## 2025-10-19 - Latent Node Tooltips and Schema Metadata
**Learning:** Adding descriptive node documentation and explicit parameter `tooltip` arguments directly within `io.Schema` definitions in node extensions improves UI accessibility, parameter discoverability, and contextual hover help across ComfyUI frontends.
**Action:** Always include node `description` and input/output parameter `tooltip` strings when defining schemas for `io.ComfyNode` classes. Validate using AST parsing in `tests-unit/`.
