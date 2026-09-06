import sys
from types import ModuleType
from unittest.mock import MagicMock

def mock_module(name):
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod

av_mod = mock_module("av")
av_container = mock_module("av.container")
av_container.InputContainer = MagicMock
av_sub = mock_module("av.subtitles")
av_sub_stream = mock_module("av.subtitles.stream")
av_sub_stream.SubtitleStream = MagicMock
einops_mod = mock_module("einops")
einops_mod.rearrange = MagicMock()
torch_mod = mock_module("torch")
torch_mod.randint = MagicMock()
torch_mod.Tensor = MagicMock
mock_module("numpy")
tqdm_mod = mock_module("tqdm")
tqdm_mod.tqdm = MagicMock
pil_mod = mock_module("PIL")
pil_image = mock_module("PIL.Image")
pil_image.Image = MagicMock
pil_png = mock_module("PIL.PngImagePlugin")
pil_png.PngInfo = MagicMock
pil_mod.Image = pil_image
pil_mod.PngImagePlugin = pil_png

from comfy_extras.nodes_hypertile import HyperTile


def test_hypertile_schema_metadata():
    schema = HyperTile.define_schema()

    assert schema.node_id == "HyperTile"
    assert schema.display_name == "HyperTile"
    assert schema.category == "model_patches/unet"
    assert "Splits self-attention computation" in schema.description

    expected_aliases = {
        "hypertile",
        "tile",
        "tiling",
        "attention tile",
        "speedup",
        "memory optimization",
        "unet patch",
    }
    assert expected_aliases.issubset(set(schema.search_aliases))


def test_hypertile_schema_parameter_tooltips():
    schema = HyperTile.define_schema()

    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "model" in inputs_dict
    assert inputs_dict["model"].tooltip == "The diffusion model to apply HyperTile self-attention patching to."

    assert "tile_size" in inputs_dict
    assert inputs_dict["tile_size"].tooltip == "Target tile size in pixels for splitting attention maps."

    assert "swap_size" in inputs_dict
    assert inputs_dict["swap_size"].tooltip == "Maximum number of candidate tile division factors."

    assert "max_depth" in inputs_dict
    assert inputs_dict["max_depth"].tooltip == "Maximum depth level in UNet hierarchy to apply tiling."

    assert "scale_depth" in inputs_dict
    assert inputs_dict["scale_depth"].tooltip == "Whether to scale tile size relative to UNet layer depth."

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip == "The patched model with HyperTile attention optimization applied."


def test_hypertile_execution():
    mock_model = MagicMock()
    cloned_model = MagicMock()
    mock_model.clone.return_value = cloned_model

    result = HyperTile.execute(
        model=mock_model,
        tile_size=256,
        swap_size=2,
        max_depth=0,
        scale_depth=False,
    )

    mock_model.clone.assert_called_once()
    cloned_model.set_model_attn1_patch.assert_called_once()
    cloned_model.set_model_attn1_output_patch.assert_called_once()
    assert result == (cloned_model,)
