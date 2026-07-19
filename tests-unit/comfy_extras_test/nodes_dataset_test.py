import pytest
import os
import tempfile
import torch
from unittest.mock import patch

import comfy.utils
from comfy_extras.nodes_dataset import (
    LoadImageDataSetFromFolderNode,
    LoadImageTextDataSetFromFolderNode,
    SaveImageDataSetToFolderNode,
    SaveImageTextDataSetToFolderNode,
    SaveTrainingDataset,
    LoadTrainingDataset
)


class TestSafeJoin:
    def test_safe_join_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = os.path.abspath(temp_dir)
            nested = comfy.utils.safe_join(base, "nested")
            assert nested == os.path.join(base, "nested")

            nested_dir_create = comfy.utils.safe_join(base, "nested_create", create_dir=True)
            assert os.path.isdir(nested_dir_create)

    def test_safe_join_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = os.path.abspath(temp_dir)
            with pytest.raises(ValueError, match="Path traversal detected"):
                comfy.utils.safe_join(base, "../traversal")

            with pytest.raises(ValueError, match="Path traversal detected"):
                comfy.utils.safe_join(base, "nested/../../outside")


class TestDatasetNodesSecurity:
    @patch("folder_paths.get_input_directory")
    def test_load_image_dataset_traversal(self, mock_get_input_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_input_dir.return_value = temp_dir

            with pytest.raises(ValueError, match="Path traversal detected"):
                LoadImageDataSetFromFolderNode.execute("../outside")

    @patch("folder_paths.get_input_directory")
    def test_load_image_text_dataset_traversal(self, mock_get_input_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_input_dir.return_value = temp_dir

            with pytest.raises(ValueError, match="Path traversal detected"):
                LoadImageTextDataSetFromFolderNode.execute("../outside")

    @patch("folder_paths.get_output_directory")
    def test_save_image_dataset_traversal(self, mock_get_output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_output_dir.return_value = temp_dir

            images = [torch.rand(1, 64, 64, 3)]
            with pytest.raises(ValueError, match="Path traversal detected"):
                SaveImageDataSetToFolderNode.execute(images, ["../outside"], ["image"])

    @patch("folder_paths.get_output_directory")
    def test_save_image_text_dataset_traversal(self, mock_get_output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_output_dir.return_value = temp_dir

            images = [torch.rand(1, 64, 64, 3)]
            texts = ["caption"]
            with pytest.raises(ValueError, match="Path traversal detected"):
                SaveImageTextDataSetToFolderNode.execute(images, texts, ["../outside"], ["image"])

    @patch("folder_paths.get_output_directory")
    def test_save_training_dataset_traversal(self, mock_get_output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_output_dir.return_value = temp_dir

            latents = [{"samples": torch.rand(1, 4, 8, 8)}]
            conditioning = [[torch.rand(1, 77, 768)]]

            with pytest.raises(ValueError, match="Path traversal detected"):
                SaveTrainingDataset.execute(latents, conditioning, ["../outside"], [1000])

    @patch("folder_paths.get_output_directory")
    def test_load_training_dataset_traversal(self, mock_get_output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_output_dir.return_value = temp_dir

            with pytest.raises(ValueError, match="Path traversal detected"):
                LoadTrainingDataset.execute("../outside")
