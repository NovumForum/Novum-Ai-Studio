from comfy_extras.nodes_compositing import PorterDuffImageComposite, SplitImageWithAlpha, JoinImageWithAlpha

def test_porter_duff_schema():
    schema = PorterDuffImageComposite.define_schema()
    assert schema.description == "Composite two images using Porter-Duff alpha blending operations."

    inputs = {inp.id: inp.tooltip for inp in schema.inputs}
    assert inputs["source"] == "The foreground/source image to composite."
    assert inputs["source_alpha"] == "The alpha mask for the source image, defining its transparency. If omitted/white, the source is fully opaque."
    assert inputs["destination"] == "The background/destination image to composite onto."
    assert inputs["destination_alpha"] == "The alpha mask for the destination image, defining its transparency."
    assert inputs["mode"] == "The Porter-Duff compositing operator defining how source and destination colors and alphas are combined."


def test_split_image_with_alpha_schema():
    schema = SplitImageWithAlpha.define_schema()
    assert schema.description == "Split an RGBA image into its RGB color channels and its alpha channel (transparency mask)."

    inputs = {inp.id: inp.tooltip for inp in schema.inputs}
    assert inputs["image"] == "The input image to extract the alpha/transparency mask from."


def test_join_image_with_alpha_schema():
    schema = JoinImageWithAlpha.define_schema()
    assert schema.description == "Combine an RGB image and a transparency mask into a single RGBA image with an alpha channel."

    inputs = {inp.id: inp.tooltip for inp in schema.inputs}
    assert inputs["image"] == "The RGB color image."
    assert inputs["alpha"] == "The mask to apply as the alpha channel (transparency mask)."
