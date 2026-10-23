from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler


def test_align_your_steps_schema_metadata():
    schema = AlignYourStepsScheduler.define_schema()

    assert schema.node_id == "AlignYourStepsScheduler"
    assert schema.display_name == "Align Your Steps Scheduler"
    assert schema.description != ""
    assert "Align Your Steps" in schema.description or "sigmas" in schema.description

    assert isinstance(schema.search_aliases, list)
    assert len(schema.search_aliases) >= 3
    assert "ays" in schema.search_aliases
    assert "align your steps" in schema.search_aliases

    # Check input tooltips
    assert len(schema.inputs) == 3
    for inp in schema.inputs:
        assert inp.tooltip is not None and len(inp.tooltip) > 0

    # Check output tooltip
    assert len(schema.outputs) == 1
    for out in schema.outputs:
        assert out.tooltip is not None and len(out.tooltip) > 0
