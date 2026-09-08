import os
import pickle
import tempfile
import pytest
import torch
import comfy.cli_args

comfy.cli_args.args.cpu = True

from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation


class MaliciousPayload:
    def __reduce__(self):
        # Malicious payload attempting arbitrary command execution
        return (os.system, ("echo VULNERABLE",))


NOISE_SCHEDULE_CONFIG = {"linear_start": 0.0001, "linear_end": 0.02, "timesteps": 1000}


def test_clip_embedding_noise_augmentation_default():
    aug = CLIPEmbeddingNoiseAugmentation(NOISE_SCHEDULE_CONFIG, max_noise_level=1000)
    assert aug.data_mean.shape == (1, 256)
    assert aug.data_std.shape == (1, 256)


def test_clip_embedding_noise_augmentation_valid_stats():
    mean = torch.zeros(256)
    std = torch.ones(256)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        stats_path = f.name

    try:
        torch.save((mean, std), stats_path)
        aug = CLIPEmbeddingNoiseAugmentation(NOISE_SCHEDULE_CONFIG, max_noise_level=1000, clip_stats_path=stats_path)
        assert torch.equal(aug.data_mean, mean[None, :])
        assert torch.equal(aug.data_std, std[None, :])
    finally:
        if os.path.exists(stats_path):
            os.remove(stats_path)


def test_clip_embedding_noise_augmentation_rejects_malicious_payload():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        stats_path = f.name

    try:
        with open(stats_path, "wb") as f_out:
            pickle.dump(MaliciousPayload(), f_out)

        with pytest.raises(Exception):
            CLIPEmbeddingNoiseAugmentation(NOISE_SCHEDULE_CONFIG, max_noise_level=1000, clip_stats_path=stats_path)
    finally:
        if os.path.exists(stats_path):
            os.remove(stats_path)
