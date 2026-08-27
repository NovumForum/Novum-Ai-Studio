from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class DummyModel:
    def __init__(self):
        self.denoise_mask_fn = None

    def clone(self):
        m = DummyModel()
        m.denoise_mask_fn = self.denoise_mask_fn
        return m

    def set_model_denoise_mask_function(self, fn):
        self.denoise_mask_fn = fn


def test_differential_diffusion_schema():
    schema = DifferentialDiffusion.define_schema()
    assert schema.node_id == "DifferentialDiffusion"
    assert schema.display_name == "Differential Diffusion"
    assert "dynamically thresholding denoise masks" in schema.description

    expected_aliases = {
        "differential diffusion",
        "inpaint gradient",
        "variable denoise strength",
        "mask denoise",
        "gradient mask",
    }
    assert set(schema.search_aliases) == expected_aliases

    input_tooltips = {inp.id: inp.tooltip for inp in schema.inputs if inp.tooltip}
    assert "model" in input_tooltips
    assert "strength" in input_tooltips

    output_tooltips = [out.tooltip for out in schema.outputs if out.tooltip]
    assert len(output_tooltips) == 1
    assert "patched model" in output_tooltips[0]


def test_differential_diffusion_execution():
    dummy_model = DummyModel()
    res = DifferentialDiffusion.execute(dummy_model, strength=0.8)

    assert res is not None
    output_model = res.args[0]
    assert output_model is not dummy_model
    assert output_model.denoise_mask_fn is not None
