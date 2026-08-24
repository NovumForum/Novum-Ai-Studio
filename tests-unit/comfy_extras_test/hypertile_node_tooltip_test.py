import sys
import asyncio
from unittest.mock import MagicMock

# Mock third-party dependencies before importing nodes if they are missing
modules_to_mock = [
    "torch",
    "einops",
    "av",
    "av.container",
    "av.subtitles",
    "av.subtitles.stream",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.PngImagePlugin",
    "scipy",
    "scipy.ndimage",
    "torchaudio",
    "torchvision",
    "tqdm",
]
for mod in modules_to_mock:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from comfy_extras.nodes_hypertile import HyperTile, HyperTileExtension, comfy_entrypoint


def test_hypertile_schema_metadata():
    schema = HyperTile.define_schema()
    assert schema.node_id == "HyperTile"
    assert schema.display_name == "HyperTile"
    assert "Optimizes UNet self-attention computation speed" in schema.description
    assert "hypertile" in schema.search_aliases
    assert "attention tiling" in schema.search_aliases
    assert "attn tile" in schema.search_aliases
    assert schema.category == "model_patches/unet"

    # Verify input tooltips
    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "model" in inputs_by_id
    assert inputs_by_id["model"].tooltip == "The UNet model to apply HyperTile attention optimization to."
    assert "tile_size" in inputs_by_id
    assert "attention tiles" in inputs_by_id["tile_size"].tooltip
    assert "swap_size" in inputs_by_id
    assert "random divisor choices" in inputs_by_id["swap_size"].tooltip
    assert "max_depth" in inputs_by_id
    assert "UNet downsampling depth" in inputs_by_id["max_depth"].tooltip
    assert "scale_depth" in inputs_by_id
    assert "scale tile dimensions" in inputs_by_id["scale_depth"].tooltip

    # Verify output tooltips
    assert len(schema.outputs) == 1
    out = schema.outputs[0]
    assert out.id == "model"
    assert out.display_name == "MODEL"
    assert "HyperTile self-attention optimizations applied" in out.tooltip


def test_hypertile_execution():
    fake_model = MagicMock()
    cloned_model = MagicMock()
    fake_model.clone.return_value = cloned_model

    res = HyperTile.execute(fake_model, tile_size=256, swap_size=2, max_depth=0, scale_depth=False)
    assert fake_model.clone.called
    assert cloned_model.set_model_attn1_patch.called
    assert cloned_model.set_model_attn1_output_patch.called
    assert res == (cloned_model,)


def test_hypertile_extension():
    ext = asyncio.run(comfy_entrypoint())
    assert isinstance(ext, HyperTileExtension)
    nodes = asyncio.run(ext.get_node_list())
    assert HyperTile in nodes
