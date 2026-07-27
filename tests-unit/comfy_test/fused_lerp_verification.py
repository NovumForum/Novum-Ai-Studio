import time
import torch


def test_fused_lerp_conditioning_average():
    # Setup inputs
    t1 = torch.randn(1, 77, 2048)
    t0 = torch.randn(1, 77, 2048)
    conditioning_to_strength = 0.35

    # Original
    tw_orig = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))

    # Refactored / Fused
    tw_fused = torch.lerp(t0, t1, conditioning_to_strength)

    # Verify mathematical equivalence
    assert torch.allclose(tw_orig, tw_fused, rtol=1e-5, atol=1e-5)


def test_fused_lerp_latent_blend():
    # Setup inputs
    samples1 = torch.randn(1, 4, 64, 64)
    samples_blended = torch.randn(1, 4, 64, 64)
    blend_factor = 0.65

    # Original
    samples_blended_orig = samples1 * blend_factor + samples_blended * (1 - blend_factor)

    # Refactored / Fused
    samples_blended_fused = torch.lerp(samples_blended, samples1, blend_factor)

    # Verify mathematical equivalence
    assert torch.allclose(samples_blended_orig, samples_blended_fused, rtol=1e-5, atol=1e-5)


def benchmark():
    t1 = torch.randn(1, 77, 2048)
    t0 = torch.randn(1, 77, 2048)
    strength = 0.35
    iterations = 5000

    # Benchmark original
    start_time = time.perf_counter()
    for _ in range(iterations):
        tw_orig = torch.mul(t1, strength) + torch.mul(t0, (1.0 - strength))
        assert tw_orig is not None
    orig_time_cond = time.perf_counter() - start_time

    # Benchmark optimized
    start_time = time.perf_counter()
    for _ in range(iterations):
        tw_fused = torch.lerp(t0, t1, strength)
        assert tw_fused is not None
    fused_time_cond = time.perf_counter() - start_time

    speedup_cond = orig_time_cond / fused_time_cond

    samples1 = torch.randn(1, 4, 64, 64)
    samples_blended = torch.randn(1, 4, 64, 64)
    blend_factor = 0.65

    # Benchmark original
    start_time = time.perf_counter()
    for _ in range(iterations):
        samples_blended_orig = samples1 * blend_factor + samples_blended * (1 - blend_factor)
        assert samples_blended_orig is not None
    orig_time_blend = time.perf_counter() - start_time

    # Benchmark optimized
    start_time = time.perf_counter()
    for _ in range(iterations):
        samples_blended_fused = torch.lerp(samples_blended, samples1, blend_factor)
        assert samples_blended_fused is not None
    fused_time_blend = time.perf_counter() - start_time

    speedup_blend = orig_time_blend / fused_time_blend

    # Use log/logging instead of print, or noqa for printing in benchmark output if run as script
    print("--------------------------------------------------")  # noqa: T201
    print("⚡ Benchmarking ConditioningAverage Optimization ⚡")  # noqa: T201
    print("--------------------------------------------------")  # noqa: T201
    print(f"Original ConditioningAverage time: {orig_time_cond:.6f}s")  # noqa: T201
    print(f"Optimized ConditioningAverage time: {fused_time_cond:.6f}s")  # noqa: T201
    print(f"Speedup: {speedup_cond:.2f}x ({((speedup_cond - 1) * 100):.1f}% faster)\n")  # noqa: T201

    print("---------------------------------------------")  # noqa: T201
    print("⚡ Benchmarking LatentBlend Optimization ⚡")  # noqa: T201
    print("---------------------------------------------")  # noqa: T201
    print(f"Original LatentBlend time: {orig_time_blend:.6f}s")  # noqa: T201
    print(f"Optimized LatentBlend time: {fused_time_blend:.6f}s")  # noqa: T201
    print(f"Speedup: {speedup_blend:.2f}x ({((speedup_blend - 1) * 100):.1f}% faster)\n")  # noqa: T201


if __name__ == "__main__":
    test_fused_lerp_conditioning_average()
    test_fused_lerp_latent_blend()
    print("✅ Assertions passed! The refactored math is identical.")  # noqa: T201
    benchmark()
