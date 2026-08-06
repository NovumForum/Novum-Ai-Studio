from comfy_extras.nodes_resolution import ResolutionSelector, AspectRatio


class TestResolutionSelector:

    def test_schema_metadata(self):
        """Verify that ResolutionSelector's schema metadata is correctly defined."""
        schema = ResolutionSelector.GET_SCHEMA()

        assert schema.node_id == "ResolutionSelector"
        assert schema.display_name == "Resolution Selector"
        assert schema.category == "utils"
        assert "calculate the optimal width and height" in schema.description

        # Verify search aliases
        expected_aliases = [
            "aspect ratio",
            "aspect",
            "dimensions",
            "resolution",
            "image size",
            "latent size",
            "megapixels",
            "empty latent size",
            "width",
            "height",
            "ratio",
        ]
        for alias in expected_aliases:
            assert alias in schema.search_aliases

    def test_schema_inputs(self):
        """Verify that input parameter schemas have proper tooltips and settings."""
        schema = ResolutionSelector.GET_SCHEMA()

        # Find inputs
        aspect_ratio_input = next(i for i in schema.inputs if i.id == "aspect_ratio")
        megapixels_input = next(i for i in schema.inputs if i.id == "megapixels")

        assert aspect_ratio_input.default == AspectRatio.SQUARE
        assert "Choose from standard aspect ratios" in aspect_ratio_input.tooltip

        assert megapixels_input.default == 1.0
        assert megapixels_input.min == 0.1
        assert megapixels_input.max == 16.0
        assert "Target total pixel count" in megapixels_input.tooltip

    def test_schema_outputs(self):
        """Verify that output parameter schemas have proper tooltips."""
        schema = ResolutionSelector.GET_SCHEMA()

        width_output = next(o for o in schema.outputs if o.id == "width")
        height_output = next(o for o in schema.outputs if o.id == "height")

        assert "Calculated target width in pixels" in width_output.tooltip
        assert "Calculated target height in pixels" in height_output.tooltip

    def test_execution_square(self):
        """Verify execution logic for square aspect ratio (1:1)."""
        result = ResolutionSelector.execute(AspectRatio.SQUARE, 1.0)
        # 1.0 MP square aspect ratio => 1024x1024
        assert result.result == (1024, 1024)

        # Check multiples of 8
        assert result.result[0] % 8 == 0
        assert result.result[1] % 8 == 0

    def test_execution_widescreen(self):
        """Verify execution logic for widescreen aspect ratio (16:9)."""
        result = ResolutionSelector.execute(AspectRatio.WIDESCREEN_H, 1.0)
        width, height = result.result

        # Total pixels should be approximately 1.0 megapixels
        # Let's ensure outputs are positive and multiple of 8
        assert width > 0
        assert height > 0
        assert width % 8 == 0
        assert height % 8 == 0

        # Aspect ratio should be approximately 16:9 (1.777)
        ratio = width / height
        assert abs(ratio - 16/9) < 0.1

    def test_execution_different_megapixels(self):
        """Verify execution logic with varying megapixel sizes."""
        for mp in [0.25, 0.5, 2.0, 4.0]:
            result = ResolutionSelector.execute(AspectRatio.STANDARD_H, mp)
            width, height = result.result
            assert width % 8 == 0
            assert height % 8 == 0
            assert width > 0
            assert height > 0
