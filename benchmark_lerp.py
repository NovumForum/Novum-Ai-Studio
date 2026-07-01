import torch
import time

def benchmark():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Typical latent shape for SDXL: [1, 4, 128, 128]
    # But usually it's batched or has multiple conds.
    # Let's try a larger tensor to see the difference.
    shape = (1, 4, 128, 128)
    uncond = torch.randn(shape, device=device)
    cond = torch.randn(shape, device=device)
    scale = 7.5

    iters = 10000

    # Warmup
    for _ in range(100):
        _ = uncond + (cond - uncond) * scale
        _ = torch.lerp(uncond, cond, scale)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        res1 = uncond + (cond - uncond) * scale
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Manual: {(end - start) * 1000 / iters:.6f} ms")

    start = time.perf_counter()
    for _ in range(iters):
        res2 = torch.lerp(uncond, cond, scale)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Lerp:   {(end - start) * 1000 / iters:.6f} ms")

    assert torch.allclose(res1, res2)

if __name__ == "__main__":
    benchmark()
