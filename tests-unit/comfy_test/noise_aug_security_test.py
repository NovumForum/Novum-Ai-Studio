import os
import tempfile
import pickle
import torch
import pytest

# Ensure CPU mode is enabled for tests to avoid CUDA initialization errors in non-GPU environments
import comfy.cli_args
comfy.cli_args.args.cpu = True

from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation


class MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


def test_clip_embedding_noise_augmentation_loads_valid_stats():
    timestep_dim = 256
    clip_mean = torch.randn(timestep_dim)
    clip_std = torch.rand(timestep_dim) + 0.1

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        stats_path = f.name
        torch.save((clip_mean, clip_std), stats_path)

    try:
        model = CLIPEmbeddingNoiseAugmentation(
            noise_schedule_config={"linear_start": 0.0001, "linear_end": 0.02},
            clip_stats_path=stats_path,
            timestep_dim=timestep_dim,
        )
        assert torch.allclose(model.data_mean[0], clip_mean)
        assert torch.allclose(model.data_std[0], clip_std)
    finally:
        if os.path.exists(stats_path):
            os.remove(stats_path)


def test_clip_embedding_noise_augmentation_rejects_malicious_payload():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        stats_path = f.name
        with open(stats_path, "wb") as pf:
            pickle.dump(MaliciousPayload(), pf)

    try:
        with pytest.raises((pickle.UnpicklingError, RuntimeError, AttributeError, ValueError)):
            CLIPEmbeddingNoiseAugmentation(
                noise_schedule_config={"linear_start": 0.0001, "linear_end": 0.02},
                clip_stats_path=stats_path,
                timestep_dim=256,
            )
    finally:
        if os.path.exists(stats_path):
            os.remove(stats_path)
