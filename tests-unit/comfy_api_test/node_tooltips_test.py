from comfy_extras.nodes_string import StringSubstring, StringContains, RegexExtract
from comfy_extras.nodes_color import ColorToRGBInt


def get_input(schema, input_id):
    for inp in schema.inputs:
        if inp.id == input_id:
            return inp
    raise ValueError(f"Input with ID {input_id} not found in schema.")


def test_string_substring_tooltips():
    schema = StringSubstring.define_schema()
    start_input = get_input(schema, "start")
    end_input = get_input(schema, "end")

    assert start_input.tooltip == "0-based start index (inclusive) of the substring"
    assert end_input.tooltip == "0-based end index (exclusive) of the substring"


def test_string_contains_tooltip():
    schema = StringContains.define_schema()
    case_sensitive_input = get_input(schema, "case_sensitive")

    assert case_sensitive_input.tooltip == "Enable for exact case matching, or disable for case-insensitive matching"


def test_regex_extract_tooltip():
    schema = RegexExtract.define_schema()
    group_index_input = get_input(schema, "group_index")

    assert group_index_input.tooltip == "Index of the regex capture group to extract. Use 0 for the entire match, or 1 for the first captured group in parentheses."


def test_color_to_rgb_int_tooltip():
    schema = ColorToRGBInt.define_schema()
    color_input = get_input(schema, "color")

    assert color_input.tooltip == "Hex color code (e.g. #ffffff) to convert into an RGB integer value."
