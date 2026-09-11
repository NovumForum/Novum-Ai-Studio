import pytest
from comfy_extras.nodes_morphology import Morphology, ImageRGBToYUV, ImageYUVToRGB


def test_morphology_schema_tooltips():
    schema = Morphology.define_schema()
    assert schema.node_id == "Morphology"
    assert schema.display_name == "Image Morphology"
    assert "mathematical morphology operations" in schema.description.lower()
    assert "morphology" in schema.search_aliases
    assert "erode" in schema.search_aliases
    assert "dilate" in schema.search_aliases

    # Check input tooltips
    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_dict
    assert inputs_dict["image"].tooltip is not None
    assert "operation" in inputs_dict
    assert inputs_dict["operation"].tooltip is not None
    assert "kernel_size" in inputs_dict
    assert inputs_dict["kernel_size"].tooltip is not None

    # Check output tooltips
    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip is not None


def test_image_rgb_to_yuv_schema_tooltips():
    schema = ImageRGBToYUV.define_schema()
    assert schema.node_id == "ImageRGBToYUV"
    assert schema.display_name == "RGB to YUV (YCbCr)"
    assert "convert an rgb image into separate y" in schema.description.lower()
    assert "rgb to yuv" in schema.search_aliases

    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs_dict
    assert inputs_dict["image"].tooltip is not None

    assert len(schema.outputs) == 3
    for out in schema.outputs:
        assert out.tooltip is not None


def test_image_yuv_to_rgb_schema_tooltips():
    schema = ImageYUVToRGB.define_schema()
    assert schema.node_id == "ImageYUVToRGB"
    assert schema.display_name == "YUV (YCbCr) to RGB"
    assert "combine separate y" in schema.description.lower()
    assert "yuv to rgb" in schema.search_aliases

    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "Y" in inputs_dict
    assert inputs_dict["Y"].tooltip is not None
    assert "U" in inputs_dict
    assert inputs_dict["U"].tooltip is not None
    assert "V" in inputs_dict
    assert inputs_dict["V"].tooltip is not None

    assert len(schema.outputs) == 1
    assert schema.outputs[0].tooltip is not None
