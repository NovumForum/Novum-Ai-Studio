# Bolt's Journal

## 2025-02-14 - Optimized string_to_seed using C-Extension
**Learning:** In critical, frequently-called code paths like parameter name hashing, calculating CRC32 manually in a Python loop is a significant bottleneck. Replacing it with standard C-extension libraries (`binascii.crc32`) with a safe fast-path gives a ~250x performance speedup on CPU.
**Action:** Always look for manual implementations of standard mathematical or hash functions in Python (e.g., manual loops) and replace them with built-in modules or optimized C-extensions like `binascii` or `hashlib`.
