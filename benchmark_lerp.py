import torch
import time

def benchmark():
    # Shapes typically found in attention outputs
    # NAG is applied to attention output, which could be (batch, seq_len, dim)
    # e.g. (1, 4096, 64) or larger for high res
    shape = (1, 4096, 320)
    a = torch.randn(shape)
    b = torch.randn(shape)
    alpha = 0.5

    # Warmup
    for _ in range(100):
        _ = a * alpha + b * (1.0 - alpha)
        _ = torch.lerp(b, a, alpha)

    n_iter = 1000

    start = time.perf_counter()
    for _ in range(n_iter):
        res1 = a * alpha + b * (1.0 - alpha)
    end = time.perf_counter()
    time_manual = (end - start) / n_iter

    start = time.perf_counter()
    for _ in range(n_iter):
        res2 = torch.lerp(b, a, alpha)
    end = time.perf_counter()
    time_lerp = (end - start) / n_iter

    print(f"Manual: {time_manual:.8f}s")
    print(f"Lerp:   {time_lerp:.8f}s")
    print(f"Speedup: {time_manual / time_lerp:.2f}x")

    # Verify correctness
    torch.testing.assert_close(res1, res2)
    print("Verification successful!")

if __name__ == "__main__":
    benchmark()
