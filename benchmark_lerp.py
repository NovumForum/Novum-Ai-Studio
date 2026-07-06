import torch
import time

def benchmark_lerp(shape, device='cpu', dtype=torch.float32, iterations=1000):
    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    w = 0.3

    # Warmup
    for _ in range(10):
        _ = a + (b - a) * w
        _ = (1 - w) * a + w * b
        _ = torch.lerp(a, b, w)

    # Method 1: a + (b - a) * w
    start = time.perf_counter()
    for _ in range(iterations):
        res1 = a + (b - a) * w
    end = time.perf_counter()
    t1 = end - start

    # Method 2: (1 - w) * a + w * b
    start = time.perf_counter()
    for _ in range(iterations):
        res2 = (1 - w) * a + w * b
    end = time.perf_counter()
    t2 = end - start

    # Method 3: torch.lerp
    start = time.perf_counter()
    for _ in range(iterations):
        res3 = torch.lerp(a, b, w)
    end = time.perf_counter()
    t3 = end - start

    print(f"Shape: {shape}, Device: {device}, Dtype: {dtype}")
    print(f"  a + (b - a) * w: {t1:.6f}s")
    print(f"  (1 - w) * a + w * b: {t2:.6f}s")
    print(f"  torch.lerp: {t3:.6f}s")
    print(f"  Speedup (lerp vs a+(b-a)*w): {t1/t3:.2f}x")
    print(f"  Speedup (lerp vs (1-w)a+wb): {t2/t3:.2f}x")

if __name__ == "__main__":
    # Latent shape (e.g. 512x512 SD1.5)
    benchmark_lerp((1, 4, 64, 64))
    # Latent shape (e.g. 1024x1024 SDXL)
    benchmark_lerp((1, 4, 128, 128))
    # Conditioning shape (SDXL)
    benchmark_lerp((1, 77, 2048))
    # Flux latent shape (typical)
    benchmark_lerp((1, 16, 128, 128))
    # Large tensors
    benchmark_lerp((2048, 2048))
