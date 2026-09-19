import unittest.mock
import torch

# Mock torch.cuda for CPU test environment
unittest.mock.patch("torch.cuda.is_available", return_value=False).start()
unittest.mock.patch("torch.cuda.current_device", return_value=0).start()
unittest.mock.patch("torch.cuda.get_device_properties", return_value=unittest.mock.MagicMock(total_memory=8 * 1024**3)).start()
unittest.mock.patch("torch.cuda.memory_stats", return_value={"reserved_bytes.all.current": 0, "allocated_bytes.all.current": 0}).start()
unittest.mock.patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)).start()

from comfy_extras.nodes_image_compare import ImageCompare


def test_image_compare_schema():
    schema = ImageCompare.define_schema()
    assert schema.node_id == "ImageCompare"
    assert schema.display_name == "Image Compare"
    assert schema.description == "Compares two images side by side with an interactive slider."
    assert schema.category == "image"
    assert schema.search_aliases == [
        "compare images",
        "image slider",
        "before after",
        "visual comparison",
        "side by side",
    ]

    inputs = {inp.id: inp for inp in schema.inputs}
    assert "image_a" in inputs
    assert inputs["image_a"].tooltip == "The first image (Image A / Before) to compare."

    assert "image_b" in inputs
    assert inputs["image_b"].tooltip == "The second image (Image B / After) to compare."

    assert "compare_view" in inputs
    assert inputs["compare_view"].tooltip == "Interactive visual comparison widget UI control."


def test_image_compare_execution():
    img_a = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    img_b = torch.ones((1, 64, 64, 3), dtype=torch.float32)

    with unittest.mock.patch("nodes.PreviewImage.save_images") as mock_save:
        mock_save.side_effect = lambda img, prefix: {"ui": {"images": [{"filename": f"{prefix}_test.png"}]}}

        res = ImageCompare.execute(image_a=img_a, image_b=img_b)

        assert res.ui is not None
        ui_res = res.ui
        assert len(ui_res["a_images"]) == 1
        assert len(ui_res["b_images"]) == 1
        assert ui_res["a_images"][0]["filename"] == "comfy.compare.a_test.png"
        assert ui_res["b_images"][0]["filename"] == "comfy.compare.b_test.png"
