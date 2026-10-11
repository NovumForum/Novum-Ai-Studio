import ast
import os

def test_audio_nodes_schema_metadata():
    filepath = os.path.join(os.path.dirname(__file__), "../../comfy_extras/nodes_audio.py")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="nodes_audio.py")

    node_classes = [
        "EmptyLatentAudio",
        "ConditioningStableAudio",
        "VAEEncodeAudio",
        "VAEDecodeAudio",
        "VAEDecodeAudioTiled",
        "SaveAudio",
        "SaveAudioMP3",
        "SaveAudioOpus",
        "PreviewAudio",
        "LoadAudio",
        "RecordAudio",
        "TrimAudioDuration",
        "SplitAudioChannels",
        "JoinAudioChannels",
        "AudioConcat",
        "AudioMerge",
        "AudioAdjustVolume",
        "EmptyAudio",
        "AudioEqualizer3Band",
    ]

    found_classes = set()

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name in node_classes:
            found_classes.add(stmt.name)
            # Find define_schema method
            schema_method = None
            for item in stmt.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    schema_method = item
                    break

            assert schema_method is not None, f"define_schema not found in {stmt.name}"

            # Look for IO.Schema call
            schema_call = None
            for node in ast.walk(schema_method):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "Schema":
                    schema_call = node
                    break

            assert schema_call is not None, f"IO.Schema call not found in {stmt.name}.define_schema"

            # Check keywords in Schema call
            keywords = {kw.arg: kw.value for kw in schema_call.keywords if kw.arg}
            assert "description" in keywords, f"Missing 'description' in {stmt.name}.define_schema"
            assert "search_aliases" in keywords, f"Missing 'search_aliases' in {stmt.name}.define_schema"

            # Check that inputs exist and contain tooltips
            assert "inputs" in keywords, f"Missing 'inputs' in {stmt.name}.define_schema"
            inputs_list = keywords["inputs"]
            assert isinstance(inputs_list, ast.List), f"'inputs' should be a list in {stmt.name}"

            for input_item in inputs_list.elts:
                if isinstance(input_item, ast.Call):
                    input_kws = {kw.arg: kw.value for kw in input_item.keywords if kw.arg}
                    assert "tooltip" in input_kws, f"Missing 'tooltip' in input parameter of {stmt.name}"

    assert found_classes == set(node_classes), f"Missing classes: {set(node_classes) - found_classes}"
