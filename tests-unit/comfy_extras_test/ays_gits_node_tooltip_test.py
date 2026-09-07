from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
from comfy_extras.nodes_gits import GITSScheduler


def test_align_your_steps_scheduler_schema():
    schema = AlignYourStepsScheduler.define_schema()
    assert schema.node_id == "AlignYourStepsScheduler"
    assert schema.display_name == "Align Your Steps Scheduler"
    assert "Align Your Steps approach" in schema.description
    assert "align your steps" in schema.search_aliases
    assert "nvidia scheduler" in schema.search_aliases

    input_names = [inp.name for inp in schema.inputs]
    assert "model_type" in input_names
    assert "steps" in input_names
    assert "denoise" in input_names

    for inp in schema.inputs:
        assert inp.tooltip is not None and len(inp.tooltip) > 0

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip is not None and len(schema.outputs[0].tooltip) > 0


def test_gits_scheduler_schema():
    schema = GITSScheduler.define_schema()
    assert schema.node_id == "GITSScheduler"
    assert schema.display_name == "GITS Scheduler"
    assert "Generative Image Transformation Steps" in schema.description
    assert "gits" in schema.search_aliases

    input_names = [inp.name for inp in schema.inputs]
    assert "coeff" in input_names
    assert "steps" in input_names
    assert "denoise" in input_names

    for inp in schema.inputs:
        assert inp.tooltip is not None and len(inp.tooltip) > 0

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip is not None and len(schema.outputs[0].tooltip) > 0
