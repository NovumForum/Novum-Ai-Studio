## 2026-03-30 - Standardizing Schema Metadata & Tooltips for Compositing Nodes
**Learning:** Node schema tooltips in ComfyUI extensions directly feed node search and contextual UI popovers, significantly reducing ambiguity for complex operations like Porter-Duff compositing.
**Action:** When defining `io.Schema` for ComfyUI nodes, always provide `description` along with explicit `tooltip` arguments for all inputs and outputs.
