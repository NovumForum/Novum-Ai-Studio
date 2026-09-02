import os
import tempfile
import pickle
import pytest
import torch
from comfy_extras.nodes_dataset import LoadTrainingDataset, SaveTrainingDataset
import folder_paths


class MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


def test_load_training_dataset_valid(tmp_path):
    # Set folder_paths output directory to temp path
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    folder_paths.get_output_directory = lambda: str(output_dir)

    dataset_dir = output_dir / "my_dataset"
    dataset_dir.mkdir()

    valid_latents = [{"samples": torch.randn(1, 4, 32, 32)}]
    valid_conditioning = [[torch.randn(1, 77, 768)]]

    shard_data = {
        "latents": valid_latents,
        "conditioning": valid_conditioning,
    }

    shard_path = dataset_dir / "shard_0000.pkl"
    with open(shard_path, "wb") as f:
        torch.save(shard_data, f)

    output = LoadTrainingDataset.execute("my_dataset")
    latents, conditioning = output[0], output[1]

    assert len(latents) == 1
    assert len(conditioning) == 1
    assert torch.equal(latents[0]["samples"], valid_latents[0]["samples"])


def test_load_training_dataset_rejects_malicious_pickle(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    folder_paths.get_output_directory = lambda: str(output_dir)

    dataset_dir = output_dir / "bad_dataset"
    dataset_dir.mkdir()

    shard_path = dataset_dir / "shard_0000.pkl"
    with open(shard_path, "wb") as f:
        pickle.dump(MaliciousPayload(), f)

    with pytest.raises(Exception) as exc_info:
        LoadTrainingDataset.execute("bad_dataset")

    # PyTorch weights_only=True raises RuntimeError or UnpicklingError on arbitrary objects
    assert "Unsupported class" in str(exc_info.value) or "weights_only" in str(exc_info.value) or "Weights only" in str(exc_info.value) or "pickle" in str(exc_info.value).lower() or isinstance(exc_info.value, (RuntimeError, pickle.UnpicklingError))
