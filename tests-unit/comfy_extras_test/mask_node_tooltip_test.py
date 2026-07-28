from unittest.mock import patch, MagicMock

# Mock nodes module to prevent parent imports failing / CUDA initialization
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

with patch.dict('sys.modules', {'nodes': mock_nodes}):
    from comfy_extras.nodes_mask import GrowMask, FeatherMask, MaskComposite


class TestMaskNodeTooltips:
    def test_grow_mask_schema(self):
        schema = GrowMask.GET_SCHEMA()
        assert schema.description == "Expand or shrink a mask by a specified number of pixels. Use positive values to grow, negative to shrink."

        inputs = {inp.id: inp for inp in schema.inputs}
        assert inputs["mask"].tooltip == "The input mask to expand or shrink."
        assert inputs["expand"].tooltip == "Number of pixels to grow (positive values) or shrink (negative values) the mask."
        assert inputs["tapered_corners"].tooltip == "When enabled, rounds/tapers the corners of the mask during expansion or erosion."

    def test_feather_mask_schema(self):
        schema = FeatherMask.GET_SCHEMA()
        assert schema.description == "Soften the edges of a mask by creating a linear fade-out gradient inward from the borders."

        inputs = {inp.id: inp for inp in schema.inputs}
        assert inputs["mask"].tooltip == "The input mask to soften/feather."
        assert inputs["left"].tooltip == "Amount of pixels to fade along the left edge."
        assert inputs["top"].tooltip == "Amount of pixels to fade along the top edge."
        assert inputs["right"].tooltip == "Amount of pixels to fade along the right edge."
        assert inputs["bottom"].tooltip == "Amount of pixels to fade along the bottom edge."

    def test_mask_composite_schema(self):
        schema = MaskComposite.GET_SCHEMA()
        assert schema.description == "Position and blend two masks together using logical or arithmetic blending operations."

        inputs = {inp.id: inp for inp in schema.inputs}
        assert inputs["destination"].tooltip == "The background mask that serves as the base canvas."
        assert inputs["source"].tooltip == "The foreground mask to place and blend onto the destination."
        assert inputs["x"].tooltip == "Horizontal offset (in pixels) for placing the source mask onto the destination."
        assert inputs["y"].tooltip == "Vertical offset (in pixels) for placing the source mask onto the destination."
        assert inputs["operation"].tooltip == "The mathematical or logical operation used to combine the overlapping parts of the masks."
