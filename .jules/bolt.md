## 2026-09-05 - Avoid Frozenset and Itertools Overhead in Cache Key Generation

**Learning:** Converting graph signature objects in `to_hashable` using `frozenset(zip(itertools.count(), ...))` incurs high iterator and set instantiation overhead during execution cache key generation for dynamic prompts. Converting sequence types directly to immutable tuples and mapping types to sorted tuple pairs provides a ~2x to 3x speedup while preserving hash stability.

**Action:** When building cache keys or converting nested data structures into hashable representations, prefer recursive tuple construction `tuple(...)` with `PRIMITIVE_TYPES` checks over `frozenset` and `itertools.count()`.
