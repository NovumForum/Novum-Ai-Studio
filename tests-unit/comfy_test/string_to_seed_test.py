import binascii
import time
import pytest
from comfy.utils import string_to_seed

def string_to_seed_manual_fallback(data):
    """The original unoptimized manual loop implementation for reference/fallback validation."""
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

def test_string_to_seed_correctness_ascii():
    # Test typical ASCII inputs of varying lengths
    test_cases = [
        "",
        "hello",
        "diffusion_model.norm.weight",
        "a",
        "A",
        "another_longer_ascii_string_with_numbers_1234567890_and_special_chars_!@#",
    ]
    for tc in test_cases:
        expected = string_to_seed_manual_fallback(tc)
        actual = string_to_seed(tc)
        assert actual == expected, f"Failed for ASCII case: {tc!r}"

def test_string_to_seed_correctness_unicode():
    # Test unicode inputs (non-ASCII strings) which must fall back to the manual loop
    # to maintain backward compatibility since their UTF-8 byte representation differs from unicode ord() sequence.
    test_cases = [
        "🔥",
        "hello🔥",
        "µ",
        "ComfyUI-高性能-⚡",
    ]
    for tc in test_cases:
        expected = string_to_seed_manual_fallback(tc)
        actual = string_to_seed(tc)
        assert actual == expected, f"Failed for unicode case: {tc!r}"

def test_string_to_seed_bytes_and_bytearray():
    # Test bytes and bytearrays
    test_cases = [
        b"hello",
        b"diffusion_model.norm.weight",
        bytearray(b"hello bytearray"),
    ]
    for tc in test_cases:
        expected = string_to_seed_manual_fallback(tc)
        actual = string_to_seed(tc)
        assert actual == expected, f"Failed for bytes/bytearray case: {tc!r}"

def test_string_to_seed_fallback_other_types():
    # Test other iterables like list and tuple
    test_cases = [
        [1, 2, 3],
        (1, 2, 3),
        [120, 121, 122],
    ]
    for tc in test_cases:
        expected = string_to_seed_manual_fallback(tc)
        actual = string_to_seed(tc)
        assert actual == expected, f"Failed for iterable case: {tc!r}"

def test_string_to_seed_performance_benchmark():
    # Benchmark to verify that the fast path is indeed highly optimized on CPU.
    # ASCII string represents the most common usage in the codebase.
    s = "diffusion_model.norm.weight"

    # Warm up
    for _ in range(100):
        string_to_seed(s)
        string_to_seed_manual_fallback(s)

    start_time = time.perf_counter()
    for _ in range(10000):
        string_to_seed_manual_fallback(s)
    manual_duration = time.perf_counter() - start_time

    start_time = time.perf_counter()
    for _ in range(10000):
        string_to_seed(s)
    fast_duration = time.perf_counter() - start_time

    speedup = manual_duration / max(fast_duration, 1e-9)
    print(f"\nManual duration: {manual_duration:.6f}s | Fast duration: {fast_duration:.6f}s | Speedup: {speedup:.2f}x")

    # Assert a clear measurable speedup on CPU
    assert speedup > 5.0, f"Expected a speedup of at least 5x, but got {speedup:.2f}x"
