import pytest
from comfy_extras.nodes_morphology import Morphology, ImageRGBToYUV, ImageYUVToRGB


class TestMorphologyNodeTooltips:
    def test_morphology_schema(self):
        schema = Morphology.define_schema()
        assert schema.node_id == "Morphology"
        assert schema.display_name == "ImageMorphology"
        assert "mathematical morphology operations" in schema.description
        assert "erode" in schema.search_aliases
        assert "dilate" in schema.search_aliases
        assert "morphology" in schema.search_aliases

        # Test input tooltips
        inputs = {inp.id: inp for inp in schema.inputs}
        assert "image" in inputs
        assert "operation" in inputs
        assert "kernel_size" in inputs

        assert "input image" in inputs["image"].tooltip.lower()
        assert "morphology operation" in inputs["operation"].tooltip.lower()
        assert "kernel" in inputs["kernel_size"].tooltip.lower()

        # Test output tooltips
        assert len(schema.outputs) == 1
        assert "processed image" in schema.outputs[0].tooltip.lower()

    def test_image_rgb_to_yuv_schema(self):
        schema = ImageRGBToYUV.define_schema()
        assert schema.node_id == "ImageRGBToYUV"
        assert schema.display_name == "Image RGB to YUV"
        assert "YCbCr" in schema.description
        assert "rgb to yuv" in schema.search_aliases

        # Test input tooltips
        inputs = {inp.id: inp for inp in schema.inputs}
        assert "image" in inputs
        assert "input RGB image" in inputs["image"].tooltip

        # Test output tooltips
        outputs = {out.display_name: out for out in schema.outputs}
        assert "Y" in outputs
        assert "U" in outputs
        assert "V" in outputs
        assert "luminance" in outputs["Y"].tooltip.lower()
        assert "blue-difference" in outputs["U"].tooltip.lower()
        assert "red-difference" in outputs["V"].tooltip.lower()

    def test_image_yuv_to_rgb_schema(self):
        schema = ImageYUVToRGB.define_schema()
        assert schema.node_id == "ImageYUVToRGB"
        assert schema.display_name == "Image YUV to RGB"
        assert "Reconstructs an RGB image" in schema.description
        assert "yuv to rgb" in schema.search_aliases

        # Test input tooltips
        inputs = {inp.id: inp for inp in schema.inputs}
        assert "Y" in inputs
        assert "U" in inputs
        assert "V" in inputs
        assert "luminance" in inputs["Y"].tooltip.lower()
        assert "blue-difference" in inputs["U"].tooltip.lower()
        assert "red-difference" in inputs["V"].tooltip.lower()

        # Test output tooltips
        assert len(schema.outputs) == 1
        assert "reconstructed RGB image" in schema.outputs[0].tooltip
