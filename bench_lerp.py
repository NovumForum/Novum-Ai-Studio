import torch
import time

def manual_lerp(a, b, w):
    return a + (b - a) * w

def manual_lerp_v2(a, b, w):
    return (1 - w) * a + w * b

def bench(shape, iterations=1000):
    a = torch.randn(shape)
    b = torch.randn(shape)
    w = 0.7

    # Warmup
    for _ in range(10):
        manual_lerp(a, b, w)
        torch.lerp(a, b, w)

    start = time.perf_counter()
    for _ in range(iterations):
        res1 = manual_lerp(a, b, w)
    end = time.perf_counter()
    manual_time = end - start

    start = time.perf_counter()
    for _ in range(iterations):
        res2 = torch.lerp(a, b, w)
    end = time.perf_counter()
    lerp_time = end - start

    print(f"Shape: {shape}")
    print(f"Manual lerp: {manual_time:.6f}s")
    print(f"Torch lerp:  {lerp_time:.6f}s")
    print(f"Speedup: {manual_time / lerp_time:.2f}x")

    # Accuracy check
    assert torch.allclose(res1, res2)

if __name__ == "__main__":
    # Latent shape (batch, channels, h, w)
    bench((1, 4, 64, 64))
    # Conditioning shape (batch, tokens, dim)
    bench((1, 77, 768))
    # Larger batch
    bench((4, 4, 128, 128))
