## 2026-03-30 - Python Sequence Hashing Optimization in Cache Key Generation

**Learning:** Converting sequence inputs in execution cache key generation (`to_hashable`) directly to Python `tuple`s instead of constructing `frozenset(zip(itertools.count(), ...))` eliminates heavy iterator and set overhead while maintaining exact sequence order and hashability.
**Action:** When converting ordered sequences to hashable representations in Python, use tuples `tuple(to_hashable(i) for i in seq)` instead of indexed `frozenset`s.
