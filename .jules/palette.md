# Palette's Journal - UX Learnings

## 2025-05-18 - Node Schema Documentation and Tooltips in Core Nodes
**Learning:** Core node schemas in `nodes.py` (like `VAEEncode` and `VAEDecode`) are frequently used by beginners, but when input and output tooltips are missing, the UI gives no immediate guidance on data formats or roles in the graph. Using AST parsing for unit testing node schemas allows tests to run instantly without requiring heavy runtime frameworks like PyTorch or CUDA.
**Action:** When improving UI discoverability and guidance for core nodes, provide concise `DESCRIPTION`, `OUTPUT_TOOLTIPS`, and input parameter `tooltip` metadata, and verify with fast AST-based pytest tests.
