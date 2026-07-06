## 2026-07-06 - [Path Traversal and Insecure Deserialization]
**Vulnerability:** Path traversal in `get_annotated_filepath` and several dataset nodes allowed accessing files outside intended directories via `../` or absolute paths. `LoadTrainingDataset` used `torch.load` without restriction, risking RCE.
**Learning:** Core path resolution functions often lacked validation. Ad-hoc checks like `if '..' in filename` are insufficient compared to absolute path comparison using `os.path.commonpath`.
**Prevention:** Use the centralized `comfy.utils.safe_join` for all path joining involving user input. Always set `weights_only=True` in `torch.load` when loading untrusted data.
