from comfy_extras.nodes_audio import ConditioningStableAudio

def test_conditioning_stable_audio_schema():
    schema = ConditioningStableAudio.define_schema()
    schema.finalize()

    assert schema.node_id == "ConditioningStableAudio"
    assert schema.display_name == "Conditioning (Stable Audio)"
    assert "Applies start time and duration conditioning" in schema.description
    assert "audio conditioning" in schema.search_aliases
    assert "sound timing" in schema.search_aliases

    # Inputs verification
    input_ids = [inp.id for inp in schema.inputs]
    assert "positive" in input_ids
    assert "negative" in input_ids
    assert "seconds_start" in input_ids
    assert "seconds_total" in input_ids

    input_map = {inp.id: inp for inp in schema.inputs}
    assert input_map["positive"].tooltip == "The positive conditioning prompt to modify with timing parameters."
    assert input_map["negative"].tooltip == "The negative conditioning prompt to modify with timing parameters."
    assert input_map["seconds_start"].tooltip == "The starting time (offset) in seconds for audio generation."
    assert input_map["seconds_total"].tooltip == "The total duration in seconds for audio generation."

    # Outputs verification
    output_ids = [out.id for out in schema.outputs]
    assert "positive" in output_ids
    assert "negative" in output_ids

    output_map = {out.id: out for out in schema.outputs}
    assert output_map["positive"].tooltip == "The positive conditioning modified with audio timing parameters."
    assert output_map["negative"].tooltip == "The negative conditioning modified with audio timing parameters."

def test_conditioning_stable_audio_execution():
    pos = [("pos_prompt", {})]
    neg = [("neg_prompt", {})]
    res = ConditioningStableAudio.execute(pos, neg, seconds_start=10.0, seconds_total=30.0)

    res_pos, res_neg = res.args
    assert res_pos[0][1]["seconds_start"] == 10.0
    assert res_pos[0][1]["seconds_total"] == 30.0
    assert res_neg[0][1]["seconds_start"] == 10.0
    assert res_neg[0][1]["seconds_total"] == 30.0
