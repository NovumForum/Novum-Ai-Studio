import torch
import time

def manual_lerp(start, end, weight):
    return start + (end - start) * weight

def manual_lerp_2(start, end, weight):
    return end * weight + start * (1 - weight)

def benchmark():
    shape = (1, 4, 64, 64)
    iterations = 10000

    start = torch.randn(shape)
    end = torch.randn(shape)
    weight = 0.5

    # Warmup
    for _ in range(100):
        manual_lerp(start, end, weight)
        manual_lerp_2(start, end, weight)
        torch.lerp(start, end, weight)

    t0 = time.perf_counter()
    for _ in range(iterations):
        res1 = manual_lerp(start, end, weight)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(iterations):
        res2 = manual_lerp_2(start, end, weight)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    for _ in range(iterations):
        res3 = torch.lerp(start, end, weight)
    t5 = time.perf_counter()

    print(f"Manual 1 (start + (end-start)*w): {t1-t0:.6f}s")
    print(f"Manual 2 (end*w + start*(1-w)): {t3-t2:.6f}s")
    print(f"torch.lerp: {t5-t4:.6f}s")

    print(f"Speedup vs Manual 1: {(t1-t0)/(t5-t4):.2f}x")
    print(f"Speedup vs Manual 2: {(t3-t2)/(t5-t4):.2f}x")

if __name__ == "__main__":
    benchmark()
