## 2026-07-10 - Path Traversal Protection in Dataset Nodes
**Vulnerability:** New experimental dataset nodes used `os.path.join` with user-controlled strings for folder and file names, allowing arbitrary file read/write via path traversal (e.g., using `..`).
**Learning:** Experimental V3 API nodes that use `io.String.Input` for directory names require explicit server-side path validation because `String.Input` allows arbitrary user strings, unlike `Combo.Input` which restricts options to a pre-defined list.
**Prevention:** Use the centralized `safe_join` utility (introduced in `comfy/utils.py`) which uses `os.path.abspath` and `os.path.commonpath` to ensure the resolved path remains within the intended base directory.

## 2026-07-10 - Secure Deserialization of Training Shards
**Vulnerability:** The `LoadTrainingDataset` node used `torch.load` without `weights_only=True`, potentially allowing arbitrary code execution if a user loaded a malicious training shard.
**Learning:** Even internal dataset formats using `.pkl` or `.pt` extensions should default to `weights_only=True` if they only contain tensors and standard containers.
**Prevention:** Always specify `weights_only=True` in `torch.load` when the expected data structure is limited to tensors and standard Python collections.
