import os
import pickle
import tempfile
import unittest
from unittest.mock import patch
import torch
from comfy_extras.nodes_dataset import LoadTrainingDataset


class MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


class TestLoadTrainingDatasetSecurity(unittest.TestCase):
    def test_load_dataset_uses_weights_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "test_dataset")
            os.makedirs(dataset_dir)
            shard_path = os.path.join(dataset_dir, "shard_0000.pkl")

            # Create a shard file containing a custom Python object payload
            payload = {"latents": [MaliciousPayload()], "conditioning": [[]]}
            with open(shard_path, "wb") as f:
                pickle.dump(payload, f)

            with patch("folder_paths.get_output_directory", return_value=temp_dir):
                # Attempting to load with weights_only=True should raise an exception (e.g. Unsupported or UnpicklingError)
                with self.assertRaises(Exception):
                    LoadTrainingDataset.execute("test_dataset")

    def test_load_valid_tensor_dataset_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "valid_dataset")
            os.makedirs(dataset_dir)
            shard_path = os.path.join(dataset_dir, "shard_0000.pkl")

            valid_data = {
                "latents": [{"samples": torch.zeros((1, 4, 32, 32))}],
                "conditioning": [[["cond_tensor", {}]]]
            }
            with open(shard_path, "wb") as f:
                torch.save(valid_data, f)

            with patch("folder_paths.get_output_directory", return_value=temp_dir):
                output = LoadTrainingDataset.execute("valid_dataset")
                # output is io.NodeOutput(all_latents, all_conditioning)
                latents, conditioning = output.args
                self.assertEqual(len(latents), 1)
                self.assertEqual(len(conditioning), 1)


if __name__ == "__main__":
    unittest.main()
