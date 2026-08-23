import pytest
import torch
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler


def test_align_your_steps_scheduler_schema_metadata():
    schema = AlignYourStepsScheduler.define_schema()
    assert schema.node_id == "AlignYourStepsScheduler"
    assert schema.display_name == "Align Your Steps Scheduler (AYS)"
    assert "Nvidia's Align Your Steps" in schema.description
    assert "align your steps" in schema.search_aliases
    assert "nvidia ays" in schema.search_aliases
    assert "custom sigmas" in schema.search_aliases
    assert "ays" in schema.search_aliases

    # Input tooltips
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "model_type" in inputs_by_id
    assert "Target model architecture" in inputs_by_id["model_type"].tooltip

    assert "steps" in inputs_by_id
    assert "Total number of sampling steps" in inputs_by_id["steps"].tooltip

    assert "denoise" in inputs_by_id
    assert "Denoising strength fraction" in inputs_by_id["denoise"].tooltip

    # Output tooltips
    outputs = schema.outputs
    assert len(outputs) == 1
    assert outputs[0].display_name == "SIGMAS"
    assert "Calculated noise schedule sigmas" in outputs[0].tooltip


def test_align_your_steps_scheduler_execution():
    out_sd1 = AlignYourStepsScheduler.execute(model_type="SD1", steps=10, denoise=1.0)
    sigmas = out_sd1.result[0]
    assert isinstance(sigmas, torch.Tensor)
    assert len(sigmas) == 11
    assert float(sigmas[-1]) == 0.0

    out_sdxl = AlignYourStepsScheduler.execute(model_type="SDXL", steps=20, denoise=0.5)
    sigmas_sdxl = out_sdxl.result[0]
    assert isinstance(sigmas_sdxl, torch.Tensor)
    assert len(sigmas_sdxl) == 11  # 20 * 0.5 = 10 steps + 1 = 11 sigmas
    assert float(sigmas_sdxl[-1]) == 0.0

    out_zero = AlignYourStepsScheduler.execute(model_type="SVD", steps=10, denoise=0.0)
    assert len(out_zero.result[0]) == 0
