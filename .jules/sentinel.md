## 2025-05-15 - RCE and Path Traversal in Dataset Nodes
**Vulnerability:** Arbitrary code execution via insecure `torch.load` (unrestricted pickle) and path traversal via unsanitized user-provided folder names in dataset nodes.
**Learning:** Experimental V3 API nodes that accept `io.String.Input` for folder paths are particularly vulnerable to traversal if they don't explicitly validate the resulting path against a base directory. Additionally, loading user-provided datasets or model stats without `weights_only=True` allows for RCE.
**Prevention:** Always use a `safe_join` utility for user-provided paths and set `weights_only=True` in `torch.load` calls when only tensors/metadata are expected.
