import time
import binascii
from comfy.utils import string_to_seed

def string_to_seed_manual_loop(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def test_string_to_seed_correctness():
    # Standard string cases
    for s in ["", "hello", "world", "comfyui", "12345", "a" * 1000]:
        assert string_to_seed(s) == string_to_seed_manual_loop(s)
        # Check standard binascii crc32 returns the same value
        expected_crc = binascii.crc32(s.encode('utf-8'))
        assert string_to_seed(s) == expected_crc

    # Standard bytes and bytearray cases
    for b in [b"", b"hello", b"world", b"comfyui", b"12345", bytearray(b"abc")]:
        assert string_to_seed(b) == string_to_seed_manual_loop(b)
        expected_crc = binascii.crc32(b)
        assert string_to_seed(b) == expected_crc

    # Non-standard iterable cases (should trigger pure-Python fallback and still match)
    char_list = ["t", "e", "s", "t"]
    assert string_to_seed(char_list) == string_to_seed_manual_loop(char_list)
    assert string_to_seed(char_list) == binascii.crc32(b"test")

    int_list = [116, 101, 115, 116]
    assert string_to_seed(int_list) == string_to_seed_manual_loop(int_list)
    assert string_to_seed(int_list) == binascii.crc32(b"test")

    # Non-ASCII unicode characters
    # This should fall back to the manual loop and correctly yield the code point based hash,
    # preserving 100% backward compatibility for unicode strings.
    unicode_str = "🚀 comfyui 🚀"
    assert string_to_seed(unicode_str) == string_to_seed_manual_loop(unicode_str)


def test_string_to_seed_performance():
    # Benchmark standard string hashing with a larger workload
    workload = ["prefix_" + str(i) + "_longer_key_to_simulate_realistic_workload" for i in range(10000)]

    # Measure manual loop
    t0 = time.perf_counter()
    for item in workload:
        _ = string_to_seed_manual_loop(item)
    t_manual = time.perf_counter() - t0

    # Measure optimized function
    t1 = time.perf_counter()
    for item in workload:
        _ = string_to_seed(item)
    t_optimized = time.perf_counter() - t1

    speedup = t_manual / max(1e-9, t_optimized)
    print(f"\nManual loop: {t_manual:.6f}s")  # noqa: T201
    print(f"Optimized function: {t_optimized:.6f}s")  # noqa: T201
    print(f"Speedup: {speedup:.2f}x")  # noqa: T201

    # The optimized C-extension path should be substantially faster (e.g., >20x)
    assert speedup > 20.0, f"Expected speedup > 20x, but got {speedup:.2f}x"
