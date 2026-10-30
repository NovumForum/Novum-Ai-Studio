## 2025-02-18 - Fast paths for Python ABC type checks in caching

**Learning:** Checking Abstract Base Classes (`collections.abc.Mapping` / `Sequence`) via `isinstance` incurs significant overhead due to Python ABC registry inspection. Adding concrete type checks (`isinstance(obj, (dict, list, tuple))`) as fast paths before ABC fallbacks reduces hashing overhead in high-frequency functions like `to_hashable` by ~20-30%.
**Action:** When optimizing recursive object traversals or signature hashing, always check built-in concrete types (`dict`, `list`, `tuple`) before falling back to `Mapping` or `Sequence` ABCs.
