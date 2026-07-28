import torch
from comfy.weight_adapter.oft import OFTAdapter
from comfy.weight_adapter.boft import BOFTAdapter


def test_oft_fused_lerp_equivalence():
    # Test different block sizes and multipliers
    block_sizes = [4, 8, 16]
    multipliers = [0.1, 0.5, 0.9, 1.2, -0.5]

    for block_size in block_sizes:
        for multiplier in multipliers:
            # Generate mock rotation matrix R (or blocks) and identity I
            r = torch.randn(block_size, block_size)
            I = torch.eye(block_size)

            # Manual interpolation
            manual = r * multiplier + (1 - multiplier) * I

            # Fused lerp interpolation
            lerped = torch.lerp(I, r, multiplier)

            # Assert mathematical equivalence
            assert torch.allclose(manual, lerped, atol=1e-7), f"OFT lerp failed for block_size={block_size}, multiplier={multiplier}"


def test_boft_fused_lerp_equivalence():
    # Test different block sizes and strengths
    block_sizes = [4, 8, 16]
    strengths = [0.1, 0.5, 0.9, 1.2, -0.5]

    for block_size in block_sizes:
        for strength in strengths:
            # Generate mock butterfly matrix block bi and identity I
            bi = torch.randn(block_size, block_size)
            I = torch.eye(block_size)

            # Manual interpolation
            manual = bi * strength + (1 - strength) * I

            # Fused lerp interpolation
            lerped = torch.lerp(I, bi, strength)

            # Assert mathematical equivalence
            assert torch.allclose(manual, lerped, atol=1e-7), f"BOFT lerp failed for block_size={block_size}, strength={strength}"


def test_oft_adapter_g_functional():
    # Test OFTAdapter.g using mock weights and verify output
    # weights tuple: (blocks, rescale, alpha, dora_scale)
    # block_num = 2, block_size = 4
    blocks = torch.randn(2, 4, 4)
    rescale = None
    alpha = 1.0
    dora_scale = None

    adapter = OFTAdapter(loaded_keys=set(), weights=(blocks, rescale, alpha, dora_scale))
    adapter.multiplier = 0.5

    # Mock input y of shape (1, 8) matching block_num * block_size
    y = torch.randn(1, 8)

    # Calculate output using optimized g()
    out = adapter.g(y)

    assert out.shape == y.shape
    assert not torch.isnan(out).any()


def test_boft_adapter_g_functional():
    # Test BOFTAdapter.g using mock weights and verify output
    # weights tuple: (blocks, rescale, alpha, dora_scale)
    # blocks shape: (boft_m, block_num, boft_b, boft_b)
    # boft_m = 2, block_num = 2, boft_b = 4
    blocks = torch.randn(2, 2, 4, 4)
    rescale = None
    alpha = 1.0
    dora_scale = None

    adapter = BOFTAdapter(loaded_keys=set(), weights=(blocks, rescale, alpha, dora_scale))
    adapter.multiplier = 0.7

    # Mock input y of shape (1, 8) matching block_num * boft_b
    y = torch.randn(1, 8)

    # Calculate output using optimized g()
    out = adapter.g(y)

    assert out.shape == y.shape
    assert not torch.isnan(out).any()
