import pytest
import binascii
import time
import comfy.utils

def old_string_to_seed_reference(data):
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
    # List of diverse inputs to test equivalence
    inputs = [
        "hello",
        "world",
        "ComfyUI-performance-optimization",
        "", # empty string
        "a" * 1000, # longer ASCII string
        "Latin-1 characters: áéíóúñ", # non-ASCII but within Latin-1 (0-255)
        bytes([0, 1, 127, 255]),
        bytearray([10, 20, 30, 40]),
        [65, 66, 67, 68], # List of small ints (bytes representation)
        [0, 1, 2, 255],
        # Edge cases/fallbacks
        "Unicode Emoji: 🎨 Palette ⚡", # outside Latin-1
        "你好, 世界", # multi-byte Chinese unicode characters
        [0, 1, 300], # list of ints with value > 255
        ['h', 'e', 'l', 'l', 'o'], # list of characters
    ]

    for inp in inputs:
        expected = old_string_to_seed_reference(inp)
        actual = comfy.utils.string_to_seed(inp)
        assert actual == expected, f"Value mismatch for input {repr(inp)}: expected {expected}, got {actual}"

def test_string_to_seed_benchmarking():
    # Benchmark to prove the speedup
    data = "comfyui.neural_network.diffusion_model.input_blocks.4.0.weight" * 20

    # Measure optimized
    t0 = time.perf_counter()
    for _ in range(10000):
        comfy.utils.string_to_seed(data)
    t_opt = time.perf_counter() - t0

    # Measure reference
    t1 = time.perf_counter()
    for _ in range(10000):
        old_string_to_seed_reference(data)
    t_ref = time.perf_counter() - t1

    speedup = t_ref / t_opt if t_opt > 0 else float('inf')
    print(f"\n[Benchmark] Optimized: {t_opt:.6f}s | Reference: {t_ref:.6f}s | Speedup: {speedup:.1f}x")

    # Assert that optimized is faster (e.g. at least 5x faster, though realistically it's 100x+ faster)
    assert t_opt < t_ref, "Optimized version was not faster than reference!"
