# Bolt Performance Journal

## 2025-02-18 - C-Extension CRC32 Hashing Over Unicode
**Learning:** Python's built-in `binascii.crc32` accepts raw bytes. Directly encoding arbitrary unicode strings with UTF-8 alters the byte representation of non-ASCII characters (e.g. multi-byte emojis), causing different CRC32 hash results compared to Python's historical element-wise `ord(char)` loop. Using `latin-1` encoding for string inputs preserves character values exactly up to 255.
**Action:** Use Latin-1 encoding for string fast-paths with a fallback mechanism to the exact historical `string_to_seed` manual loop to maintain 100% correctness and backward compatibility.
