from unittest.mock import MagicMock, patch


class MockModule(MagicMock):
    def __getattr__(self, name):
        return MagicMock()


def test_canny_schema_metadata():
    mock_mod = MockModule()
    mocks = {
        "kornia": mock_mod,
        "kornia.filters": mock_mod,
        "comfy.model_management": mock_mod,
        "comfy.cli_args": mock_mod,
        "torch": mock_mod,
        "numpy": mock_mod,
        "PIL": mock_mod,
        "PIL.Image": mock_mod,
        "PIL.PngImagePlugin": mock_mod,
        "av": mock_mod,
        "av.container": mock_mod,
        "av.subtitles": mock_mod,
        "av.subtitles.stream": mock_mod,
        "tqdm": mock_mod,
    }

    with patch.dict("sys.modules", mocks):
        from comfy_extras.nodes_canny import Canny

        schema = Canny.define_schema()

        assert schema.node_id == "Canny"
        assert schema.display_name == "Canny Edge Detector"
        assert "Detects sharp edges" in schema.description
        assert "ControlNet guidance" in schema.description

        assert "canny edge detector" in schema.search_aliases
        assert "controlnet preprocessor" in schema.search_aliases
        assert "sketch" in schema.search_aliases
        assert "edges" in schema.search_aliases

        # Verify input tooltips
        inputs_by_id = {inp.id: inp for inp in schema.inputs}
        assert "image" in inputs_by_id
        assert inputs_by_id["image"].tooltip == "Input image tensor to perform edge detection on."

        assert "low_threshold" in inputs_by_id
        assert "Lower intensity gradient threshold" in inputs_by_id["low_threshold"].tooltip

        assert "high_threshold" in inputs_by_id
        assert "Upper intensity gradient threshold" in inputs_by_id["high_threshold"].tooltip

        # Verify output tooltips
        assert len(schema.outputs) == 1
        assert schema.outputs[0].tooltip == "Output edge map image (RGB format)."
