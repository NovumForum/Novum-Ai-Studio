from comfy_extras.nodes_ip2p import InstructPixToPixConditioning


def test_instruct_pix_to_pix_schema():
    schema = InstructPixToPixConditioning.define_schema()

    assert schema.node_id == "InstructPixToPixConditioning"
    assert schema.display_name == "Instruct PixToPix Conditioning"
    assert schema.category == "conditioning/instructpix2pix"
    assert "InstructPixToPix image editing models" in schema.description
    assert schema.search_aliases == ["instructpix2pix", "ip2p", "image editing", "instruct conditioning", "pix2pix"]

    inputs_dict = {inp.id: inp for inp in schema.inputs}
    assert "positive" in inputs_dict
    assert "negative" in inputs_dict
    assert "vae" in inputs_dict
    assert "pixels" in inputs_dict

    assert inputs_dict["positive"].tooltip == "Positive prompt conditioning to guide the edit."
    assert inputs_dict["negative"].tooltip == "Negative prompt conditioning to guide what to avoid during the edit."
    assert inputs_dict["vae"].tooltip == "VAE model used to encode the reference image pixels into latent space."
    assert inputs_dict["pixels"].tooltip == "Input reference image to be edited by InstructPixToPix."

    outputs_dict = {out.display_name: out for out in schema.outputs}
    assert "positive" in outputs_dict
    assert "negative" in outputs_dict
    assert "latent" in outputs_dict

    assert outputs_dict["positive"].display_name == "positive"
    assert outputs_dict["positive"].tooltip == "Updated positive conditioning containing concatenated latent image representations."
    assert outputs_dict["negative"].display_name == "negative"
    assert outputs_dict["negative"].tooltip == "Updated negative conditioning containing concatenated latent image representations."
    assert outputs_dict["latent"].display_name == "latent"
    assert outputs_dict["latent"].tooltip == "Empty target latent canvas initialized with matching spatial dimensions."
