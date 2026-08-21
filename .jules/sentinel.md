## 2026-03-01 - Insecure PyTorch Deserialization in LoadTrainingDataset
**Vulnerability:** `LoadTrainingDataset.execute` called `torch.load(f)` on dataset shard pickle files without `weights_only=True`, allowing arbitrary code execution when loading untrusted dataset shards.
**Learning:** PyTorch `torch.load` defaults to standard `pickle.load` deserialization unless `weights_only=True` is explicitly specified.
**Prevention:** Always pass `weights_only=True` when deserializing PyTorch files/checkpoints unless loading non-tensor arbitrary objects is strictly required and trusted.
