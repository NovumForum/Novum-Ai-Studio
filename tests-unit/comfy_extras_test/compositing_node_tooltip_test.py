import torch
from comfy_extras.nodes_compositing import (
    PorterDuffImageComposite,
    SplitImageWithAlpha,
    JoinImageWithAlpha,
)


def test_porter_duff_image_composite_schema():
    schema = PorterDuffImageComposite.define_schema()
    assert schema.node_id == "PorterDuffImageComposite"
    assert schema.display_name == "Porter-Duff Image Composite"
    assert schema.description is not None
    assert "Composites source" in schema.description
    assert "porter duff" in schema.search_aliases
    assert "matte" in schema.search_aliases

    # Inputs check
    input_names = [inp.id for inp in schema.inputs]
    assert "source" in input_names
    assert "source_alpha" in input_names
    assert "destination" in input_names
    assert "destination_alpha" in input_names
    assert "mode" in input_names

    for inp in schema.inputs:
        assert getattr(inp, "tooltip", None) is not None, f"Input {inp.id} missing tooltip"

    # Outputs check
    assert len(schema.outputs) == 2
    for out in schema.outputs:
        assert getattr(out, "display_name", None) is not None
        assert getattr(out, "tooltip", None) is not None


def test_split_image_with_alpha_schema():
    schema = SplitImageWithAlpha.define_schema()
    assert schema.node_id == "SplitImageWithAlpha"
    assert schema.display_name == "Split Image with Alpha"
    assert schema.description is not None
    assert "Separates an RGBA image tensor" in schema.description
    assert "split rgba" in schema.search_aliases

    for inp in schema.inputs:
        assert getattr(inp, "tooltip", None) is not None

    for out in schema.outputs:
        assert getattr(out, "display_name", None) is not None
        assert getattr(out, "tooltip", None) is not None


def test_join_image_with_alpha_schema():
    schema = JoinImageWithAlpha.define_schema()
    assert schema.node_id == "JoinImageWithAlpha"
    assert schema.display_name == "Join Image with Alpha"
    assert schema.description is not None
    assert "Combines RGB image channels" in schema.description
    assert "apply mask" in schema.search_aliases

    for inp in schema.inputs:
        assert getattr(inp, "tooltip", None) is not None

    for out in schema.outputs:
        assert getattr(out, "display_name", None) is not None
        assert getattr(out, "tooltip", None) is not None


def test_split_and_join_image_with_alpha_execution():
    # Test SplitImageWithAlpha with RGBA tensor (B=1, H=4, W=4, C=4)
    rgba = torch.rand((1, 4, 4, 4))
    split_res = SplitImageWithAlpha.execute(rgba)
    rgb_out, alpha_out = split_res.args

    assert rgb_out.shape == (1, 4, 4, 3)
    assert alpha_out.shape == (1, 4, 4)
    assert torch.allclose(rgb_out, rgba[:, :, :, :3])

    # Test JoinImageWithAlpha rejoining RGB and Alpha
    join_res = JoinImageWithAlpha.execute(rgb_out, alpha_out)
    joined_rgba = join_res.args[0]

    assert joined_rgba.shape == (1, 4, 4, 4)
    assert torch.allclose(joined_rgba[:, :, :, :3], rgb_out)
