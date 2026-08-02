import binascii
import pytest
import comfy.utils

def original_string_to_seed(data):
    # Reference implementation of the original manual loop
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

def test_string_to_seed_ascii():
    # Test typical model parameter keys
    test_keys = [
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
        "diffusion_model.output_blocks.2.0.weight",
        "some_random_longer_key_with_numbers_12345_and_symbols._",
        "",
    ]
    for key in test_keys:
        assert comfy.utils.string_to_seed(key) == original_string_to_seed(key)

def test_string_to_seed_bytes():
    # Test bytes & bytearrays
    test_bytes_inputs = [
        b"model.diffusion_model",
        b"hello world",
        b"",
        bytearray(b"mutable bytes")
    ]
    for key in test_bytes_inputs:
        assert comfy.utils.string_to_seed(key) == original_string_to_seed(key)

def test_string_to_seed_non_ascii_unicode():
    # Test unicode strings with non-ASCII characters to trigger and verify the fallback path
    non_ascii_inputs = [
        "hëllö",
        "こんにちは",
        "model.diffusion_model.🔑.weight",
        "café",
    ]
    for key in non_ascii_inputs:
        # Check that our optimized function still matches the exact behavior of original
        assert comfy.utils.string_to_seed(key) == original_string_to_seed(key)

def test_string_to_seed_mixed_iterables():
    # Test list of integers or mixed types (which works in the original fallback loop)
    mixed_inputs = [
        [104, 101, 108, 108, 111], # ASCII values for "hello"
        ["h", 101, "l", 108, "o"],
    ]
    for key in mixed_inputs:
        assert comfy.utils.string_to_seed(key) == original_string_to_seed(key)
