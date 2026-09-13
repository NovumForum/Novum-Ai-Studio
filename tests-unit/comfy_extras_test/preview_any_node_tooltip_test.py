from comfy_extras.nodes_preview_any import PreviewAny, NODE_DISPLAY_NAME_MAPPINGS


class UnserializableObject:
    def __str__(self):
        raise ValueError("Cannot convert to string")


def test_preview_any_metadata():
    assert hasattr(PreviewAny, "DESCRIPTION")
    assert isinstance(PreviewAny.DESCRIPTION, str)
    assert len(PreviewAny.DESCRIPTION) > 0

    assert hasattr(PreviewAny, "SEARCH_ALIASES")
    assert "debug" in PreviewAny.SEARCH_ALIASES
    assert "inspect" in PreviewAny.SEARCH_ALIASES

    input_types = PreviewAny.INPUT_TYPES()
    assert "required" in input_types
    assert "source" in input_types["required"]
    _, kwargs = input_types["required"]["source"]
    assert "tooltip" in kwargs
    assert isinstance(kwargs["tooltip"], str)
    assert len(kwargs["tooltip"]) > 0

    assert NODE_DISPLAY_NAME_MAPPINGS.get("PreviewAny") == "Preview as Text"


def test_preview_any_execution():
    node = PreviewAny()

    # String input
    res = node.main(source="hello world")
    assert res == {"ui": {"text": ("hello world",)}}

    # Numeric & boolean inputs
    res_int = node.main(source=42)
    assert res_int == {"ui": {"text": ("42",)}}

    res_bool = node.main(source=True)
    assert res_bool == {"ui": {"text": ("True",)}}

    # Dict / JSON serializable input
    res_dict = node.main(source={"key": "value"})
    assert "key" in res_dict["ui"]["text"][0]

    # None input
    res_none = node.main(source=None)
    assert res_none == {"ui": {"text": ("None",)}}

    # Unserializable object fallback
    res_unserializable = node.main(source=UnserializableObject())
    assert res_unserializable == {"ui": {"text": ("source exists, but could not be serialized.",)}}
