import os
import tempfile
import pickle
import pytest
import torch

import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation


class MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


def test_clip_embedding_noise_augmentation_weights_only():
    timestep_dim = 16
    mean = torch.randn(timestep_dim)
    std = torch.rand(timestep_dim) + 0.1
    noise_schedule_config = {"beta_schedule": "linear", "timesteps": 1000}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save((mean, std), tmp_path)

    try:
        # Should load valid tuple of tensors safely
        aug = CLIPEmbeddingNoiseAugmentation(
            noise_schedule_config=noise_schedule_config,
            max_noise_level=1000,
            clip_stats_path=tmp_path,
            timestep_dim=timestep_dim,
        )
        assert torch.allclose(aug.data_mean.squeeze(0), mean)
        assert torch.allclose(aug.data_std.squeeze(0), std)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_clip_embedding_noise_augmentation_rejects_malicious_pickle():
    noise_schedule_config = {"beta_schedule": "linear", "timesteps": 1000}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        with open(tmp_path, "wb") as f:
            pickle.dump(MaliciousPayload(), f)

    try:
        # Loading malicious payload with weights_only=True should raise error and not execute command
        with pytest.raises((Exception, AttributeError)):
            CLIPEmbeddingNoiseAugmentation(
                noise_schedule_config=noise_schedule_config,
                max_noise_level=1000,
                clip_stats_path=tmp_path,
                timestep_dim=16,
            )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
