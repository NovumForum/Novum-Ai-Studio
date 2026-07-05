import torch
import os

def test_weights_only():
    data = {
        "latents": [{"samples": torch.randn(1, 4, 64, 64)}],
        "conditioning": [[ [torch.randn(1, 77, 768), {"pooled_output": torch.randn(1, 768)}] ]]
    }

    torch.save(data, "test.pkl")

    try:
        loaded = torch.load("test.pkl", weights_only=True)
        print("Successfully loaded with weights_only=True")
    except Exception as e:
        print(f"Failed to load with weights_only=True: {e}")

    os.remove("test.pkl")

if __name__ == "__main__":
    test_weights_only()
