import os
import pickle
import tempfile
import pytest
import torch
import folder_paths
from comfy_extras.nodes_dataset import LoadTrainingDataset


class MaliciousObject:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


def test_load_training_dataset_weights_only(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        folder_name = "test_dataset"
        dataset_dir = os.path.join(tmp_dir, folder_name)
        os.makedirs(dataset_dir, exist_ok=True)

        monkeypatch.setattr(folder_paths, "get_output_directory", lambda: tmp_dir)

        # 1. Valid shard data with tensors/dicts
        valid_data = {
            "latents": [{"samples": torch.randn(1, 4, 32, 32)}],
            "conditioning": [[["cond_tensor", {}]]],
        }
        shard_path = os.path.join(dataset_dir, "shard_0000.pkl")
        with open(shard_path, "wb") as f:
            torch.save(valid_data, f)

        res = LoadTrainingDataset.execute(folder_name)
        assert len(res.args[0]) == 1
        assert len(res.args[1]) == 1

        # 2. Malicious pickle payload that attempts code execution
        malicious_path = os.path.join(dataset_dir, "shard_0001.pkl")
        with open(malicious_path, "wb") as f:
            pickle.dump(MaliciousObject(), f)

        # Expect exception due to weights_only=True blocking arbitrary class deserialization
        with pytest.raises(Exception):
            LoadTrainingDataset.execute(folder_name)
