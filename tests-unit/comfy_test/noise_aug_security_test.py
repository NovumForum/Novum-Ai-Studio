import sys
from unittest.mock import MagicMock

# Mock optional/hardware-specific modules before comfy imports
mock_aimdo = MagicMock()
sys.modules["comfy_aimdo"] = mock_aimdo
sys.modules["comfy_aimdo.torch"] = mock_aimdo
sys.modules["comfy_aimdo.model_vbar"] = mock_aimdo

import os
import pickle
import tempfile
import pytest
import torch
from unittest.mock import patch

import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation


class MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


def mock_super_init(self, *args, **kwargs):
    torch.nn.Module.__init__(self)


def test_clip_noise_aug_loads_valid_stats():
    """Verify that CLIPEmbeddingNoiseAugmentation safely loads valid tensor tuples."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create a valid clip stats file (tuple of mean and std tensors)
        valid_data = (torch.zeros(256), torch.ones(256))
        torch.save(valid_data, tmp_path)

        with patch("comfy.ldm.modules.encoders.noise_aug_modules.ImageConcatWithNoiseAugmentation.__init__", new=mock_super_init), \
             patch("comfy.ldm.modules.encoders.noise_aug_modules.Timestep", return_value=MagicMock()):
            aug = CLIPEmbeddingNoiseAugmentation(clip_stats_path=tmp_path)
            assert torch.equal(aug.data_mean, torch.zeros(256)[None, :])
            assert torch.equal(aug.data_std, torch.ones(256)[None, :])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_clip_noise_aug_rejects_malicious_payload():
    """Verify that CLIPEmbeddingNoiseAugmentation rejects malicious pickle payloads with weights_only=True."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write a malicious pickle payload to file
        payload = pickle.dumps(MaliciousPayload())
        with open(tmp_path, "wb") as f:
            f.write(payload)

        with patch("comfy.ldm.modules.encoders.noise_aug_modules.ImageConcatWithNoiseAugmentation.__init__", new=mock_super_init), \
             patch("comfy.ldm.modules.encoders.noise_aug_modules.Timestep", return_value=MagicMock()):
            with pytest.raises(Exception):
                CLIPEmbeddingNoiseAugmentation(clip_stats_path=tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
