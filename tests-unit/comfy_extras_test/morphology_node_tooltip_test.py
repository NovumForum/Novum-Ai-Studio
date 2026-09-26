import sys
from unittest.mock import MagicMock, ModuleType

# Mock torch and dependencies required for importing nodes in CPU/test environment
class MockModule(ModuleType):
    def __getattr__(self, name):
        return MagicMock()

torch = MockModule("torch")
torch.cuda = MockModule("torch.cuda")
torch.cuda.is_available = MagicMock(return_value=False)
torch.cuda.mem_get_info = MagicMock(return_value=(8000000000, 8000000000))
torch.cuda.get_device_properties = MagicMock(return_value=MagicMock(total_memory=8000000000))

sys.modules["torch"] = torch
sys.modules["torch.nn"] = MockModule("torch.nn")
sys.modules["torch.nn.functional"] = MockModule("torch.nn.functional")
sys.modules["av"] = MockModule("av")
sys.modules["av.container"] = MagicMock()
sys.modules["av.subtitles"] = MagicMock()
sys.modules["av.subtitles.stream"] = MagicMock()
sys.modules["numpy"] = MockModule("numpy")
sys.modules["numpy.dtypes"] = MagicMock()
sys.modules["PIL"] = MockModule("PIL")
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PIL.PngImagePlugin"] = MagicMock()
sys.modules["safetensors"] = MagicMock()
sys.modules["safetensors.torch"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()
sys.modules["tqdm"] = MockModule("tqdm")
sys.modules["tqdm.auto"] = MockModule("tqdm.auto")
sys.modules["einops"] = MagicMock()
sys.modules["comfy_aimdo"] = MockModule("comfy_aimdo")
sys.modules["comfy_aimdo.torch"] = MagicMock()
sys.modules["comfy_aimdo.model_vbar"] = MagicMock()
sys.modules["kornia"] = MagicMock()
sys.modules["kornia.morphology"] = MagicMock()
sys.modules["kornia.color"] = MagicMock()

from comfy_extras.nodes_morphology import Morphology, ImageRGBToYUV, ImageYUVToRGB


def test_morphology_schema_metadata():
    schema = Morphology.define_schema()
    assert schema.node_id == "Morphology"
    assert schema.display_name == "ImageMorphology"
    assert schema.category == "image/postprocessing"
    assert "morphology" in schema.search_aliases
    assert "erode" in schema.search_aliases
    assert "dilate" in schema.search_aliases
    assert "top hat" in schema.search_aliases
    assert schema.description is not None
    assert "morphological" in schema.description.lower()

    inputs = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs
    assert getattr(inputs["image"], "tooltip", None) is not None
    assert "operation" in inputs
    assert getattr(inputs["operation"], "tooltip", None) is not None
    assert "kernel_size" in inputs
    assert getattr(inputs["kernel_size"], "tooltip", None) is not None

    outputs = schema.outputs
    assert len(outputs) == 1
    assert getattr(outputs[0], "tooltip", None) is not None


def test_image_rgb_to_yuv_schema_metadata():
    schema = ImageRGBToYUV.define_schema()
    assert schema.node_id == "ImageRGBToYUV"
    assert schema.category == "image/batch"
    assert "yuv" in schema.search_aliases
    assert "color space conversion" in schema.search_aliases
    assert schema.description is not None
    assert "y (luminance)" in schema.description.lower()

    inputs = {inp.id: inp for inp in schema.inputs}
    assert "image" in inputs
    assert getattr(inputs["image"], "tooltip", None) is not None

    outputs = {getattr(out, "display_name", None): out for out in schema.outputs}
    assert "Y" in outputs
    assert getattr(outputs["Y"], "tooltip", None) is not None
    assert "U" in outputs
    assert getattr(outputs["U"], "tooltip", None) is not None
    assert "V" in outputs
    assert getattr(outputs["V"], "tooltip", None) is not None


def test_image_yuv_to_rgb_schema_metadata():
    schema = ImageYUVToRGB.define_schema()
    assert schema.node_id == "ImageYUVToRGB"
    assert schema.category == "image/batch"
    assert "yuv" in schema.search_aliases
    assert "combine channels" in schema.search_aliases
    assert schema.description is not None
    assert "y (luminance)" in schema.description.lower()

    inputs = {inp.id: inp for inp in schema.inputs}
    assert "Y" in inputs
    assert getattr(inputs["Y"], "tooltip", None) is not None
    assert "U" in inputs
    assert getattr(inputs["U"], "tooltip", None) is not None
    assert "V" in inputs
    assert getattr(inputs["V"], "tooltip", None) is not None

    outputs = schema.outputs
    assert len(outputs) == 1
    assert getattr(outputs[0], "tooltip", None) is not None


if __name__ == "__main__":
    test_morphology_schema_metadata()
    test_image_rgb_to_yuv_schema_metadata()
    test_image_yuv_to_rgb_schema_metadata()
