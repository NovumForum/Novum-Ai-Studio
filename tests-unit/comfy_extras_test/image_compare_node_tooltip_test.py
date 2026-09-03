import sys
from unittest.mock import MagicMock

sys.modules["nodes"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["av"] = MagicMock()
sys.modules["av.container"] = MagicMock()
sys.modules["av.subtitles"] = MagicMock()
sys.modules["av.subtitles.stream"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PIL.PngImagePlugin"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

from comfy_extras.nodes_image_compare import ImageCompare


def test_image_compare_schema():
    schema = ImageCompare.define_schema()

    assert schema.node_id == "ImageCompare"
    assert schema.display_name == "Image Compare"
    assert schema.description == "Compares two images side by side with a slider."
    assert schema.category == "image"
    assert schema.search_aliases == [
        "image compare",
        "side by side",
        "diff images",
        "compare images",
        "image slider",
    ]

    inputs_by_id = {inp.id: inp for inp in schema.inputs}
    assert "image_a" in inputs_by_id
    assert "image_b" in inputs_by_id
    assert "compare_view" in inputs_by_id

    assert inputs_by_id["image_a"].tooltip == "First image to compare (left / base view)."
    assert inputs_by_id["image_b"].tooltip == "Second image to compare (right / comparison view)."
    assert inputs_by_id["compare_view"].tooltip == "Interactive widget view for side-by-side comparison slider."
