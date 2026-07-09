## 2026-07-09 - Path Traversal Protection and Secure Deserialization in Dataset Nodes
**Vulnerability:** Path traversal in dataset loading/saving nodes allowing access outside input/output directories; insecure deserialization via `torch.load` in training dataset node.
**Learning:** Experimental V3 API nodes using `io.String.Input` for directory names require server-side validation via `safe_join` to prevent arbitrary filesystem access. `torch.load` must always use `weights_only=True` when loading user-provided data.
**Prevention:** Use the newly implemented `comfy.utils.safe_join` for all user-controlled path operations and prioritize `weights_only=True` for all `torch.load` calls.
