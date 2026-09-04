import sys
from unittest.mock import MagicMock

# Mock dependencies that may not be present in lightweight test environments
mock_av = MagicMock()
mock_pil = MagicMock()
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.fft', MagicMock())
sys.modules.setdefault('numpy', MagicMock())
sys.modules.setdefault('tqdm', MagicMock())
sys.modules.setdefault('PIL', mock_pil)
sys.modules.setdefault('PIL.Image', mock_pil)
sys.modules.setdefault('PIL.PngImagePlugin', mock_pil)
sys.modules.setdefault('av', mock_av)
sys.modules.setdefault('av.container', mock_av)
sys.modules.setdefault('av.subtitles', mock_av)
sys.modules.setdefault('av.subtitles.stream', mock_av)

import pytest
from comfy_extras.nodes_freelunch import FreeU, FreeU_V2


class TestFreeUNodeSchema:
    def test_freeu_schema_metadata(self):
        schema = FreeU.define_schema()
        assert schema.node_id == "FreeU"
        assert schema.display_name == "FreeU"
        assert schema.category == "model_patches/unet"
        assert schema.description is not None
        assert "UNet" in schema.description or "backbone" in schema.description
        assert schema.search_aliases is not None
        assert "freeu" in schema.search_aliases
        assert "quality boost" in schema.search_aliases

    def test_freeu_inputs_outputs_tooltips(self):
        schema = FreeU.define_schema()

        # Inputs
        input_names = [inp.id for inp in schema.inputs]
        assert "model" in input_names
        assert "b1" in input_names
        assert "b2" in input_names
        assert "s1" in input_names
        assert "s2" in input_names

        for inp in schema.inputs:
            assert inp.tooltip is not None and len(inp.tooltip) > 0

        # Outputs
        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip is not None and len(schema.outputs[0].tooltip) > 0


class TestFreeUV2NodeSchema:
    def test_freeu_v2_schema_metadata(self):
        schema = FreeU_V2.define_schema()
        assert schema.node_id == "FreeU_V2"
        assert schema.display_name == "FreeU_V2 (FreeU Version 2)"
        assert schema.category == "model_patches/unet"
        assert schema.description is not None
        assert "Version 2" in schema.description or "adaptive" in schema.description
        assert schema.search_aliases is not None
        assert "freeu_v2" in schema.search_aliases

    def test_freeu_v2_inputs_outputs_tooltips(self):
        schema = FreeU_V2.define_schema()

        # Inputs
        input_names = [inp.id for inp in schema.inputs]
        assert "model" in input_names
        assert "b1" in input_names
        assert "b2" in input_names
        assert "s1" in input_names
        assert "s2" in input_names

        for inp in schema.inputs:
            assert inp.tooltip is not None and len(inp.tooltip) > 0

        # Outputs
        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip is not None and len(schema.outputs[0].tooltip) > 0
