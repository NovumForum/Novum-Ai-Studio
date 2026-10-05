import comfy.cli_args

comfy.cli_args.args.cpu = True
import nodes


def test_vae_encode_schema():
    cls = nodes.VAEEncode
    assert hasattr(cls, "DESCRIPTION") and cls.DESCRIPTION
    assert hasattr(cls, "SEARCH_ALIASES") and isinstance(cls.SEARCH_ALIASES, list) and len(cls.SEARCH_ALIASES) > 0
    assert hasattr(cls, "OUTPUT_TOOLTIPS") and isinstance(cls.OUTPUT_TOOLTIPS, tuple) and len(cls.OUTPUT_TOOLTIPS) > 0

    inputs = cls.INPUT_TYPES()
    required = inputs.get("required", {})
    assert "pixels" in required
    assert "vae" in required
    assert "tooltip" in required["pixels"][1]
    assert "tooltip" in required["vae"][1]


def test_vae_encode_for_inpaint_schema():
    cls = nodes.VAEEncodeForInpaint
    assert hasattr(cls, "DESCRIPTION") and cls.DESCRIPTION
    assert hasattr(cls, "SEARCH_ALIASES") and isinstance(cls.SEARCH_ALIASES, list) and len(cls.SEARCH_ALIASES) > 0
    assert hasattr(cls, "OUTPUT_TOOLTIPS") and isinstance(cls.OUTPUT_TOOLTIPS, tuple) and len(cls.OUTPUT_TOOLTIPS) > 0

    inputs = cls.INPUT_TYPES()
    required = inputs.get("required", {})
    assert "pixels" in required
    assert "vae" in required
    assert "mask" in required
    assert "grow_mask_by" in required
    assert "tooltip" in required["pixels"][1]
    assert "tooltip" in required["vae"][1]
    assert "tooltip" in required["mask"][1]
    assert "tooltip" in required["grow_mask_by"][1]
