from comfy_extras.nodes_string import (
    StringConcatenate,
    StringSubstring,
    StringLength,
    CaseConverter,
    StringTrim,
    StringReplace,
    StringContains,
    StringCompare,
    RegexMatch,
    RegexExtract,
    RegexReplace,
)


def test_string_concatenate_schema_and_execution():
    schema = StringConcatenate.define_schema()
    assert schema.node_id == "StringConcatenate"
    assert schema.display_name == "Concatenate"
    assert schema.description == "Combine two text strings into one with an optional delimiter."
    assert "string" in schema.search_aliases

    # Verify input tooltips
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string_a"].tooltip == "First text string."
    assert inputs["string_b"].tooltip == "Second text string."
    assert inputs["delimiter"].tooltip == "Optional delimiter inserted between the two strings."

    # Verify execution
    out = StringConcatenate.execute("Hello", "World", ", ")
    assert out.args == ("Hello, World",)


def test_string_substring_schema_and_execution():
    schema = StringSubstring.define_schema()
    assert schema.description == "Extract a slice of a text string using start and end character indices."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to slice."
    assert inputs["start"].tooltip == "Starting character index (0-based)."
    assert inputs["end"].tooltip == "Ending character index (exclusive)."

    out = StringSubstring.execute("ComfyUI", 0, 5)
    assert out.args == ("Comfy",)


def test_string_length_schema_and_execution():
    schema = StringLength.define_schema()
    assert schema.description == "Count the total number of characters in a text string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to measure."

    out = StringLength.execute("Palette")
    assert out.args == (7,)


def test_case_converter_schema_and_execution():
    schema = CaseConverter.define_schema()
    assert schema.description == "Convert text string casing mode (e.g. UPPERCASE, lowercase, Capitalize, Title Case)."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to convert."
    assert inputs["mode"].tooltip == "Casing transformation mode to apply."

    assert CaseConverter.execute("hello world", "UPPERCASE").args == ("HELLO WORLD",)
    assert CaseConverter.execute("HELLO WORLD", "lowercase").args == ("hello world",)
    assert CaseConverter.execute("hello world", "Capitalize").args == ("Hello world",)
    assert CaseConverter.execute("hello world", "Title Case").args == ("Hello World",)


def test_string_trim_schema_and_execution():
    schema = StringTrim.define_schema()
    assert schema.description == "Strip leading and/or trailing whitespace from a text string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to trim."
    assert inputs["mode"].tooltip == "Trim mode specifying which sides to strip whitespace from."

    assert StringTrim.execute("  hello  ", "Both").args == ("hello",)
    assert StringTrim.execute("  hello  ", "Left").args == ("hello  ",)
    assert StringTrim.execute("  hello  ", "Right").args == ("  hello",)


def test_string_replace_schema_and_execution():
    schema = StringReplace.define_schema()
    assert schema.description == "Search for occurrences of a substring and replace them with new text."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to perform replacement on."
    assert inputs["find"].tooltip == "Substring to search for."
    assert inputs["replace"].tooltip == "Text string to replace found occurrences with."

    assert StringReplace.execute("foo bar foo", "foo", "baz").args == ("baz bar baz",)


def test_string_contains_schema_and_execution():
    schema = StringContains.define_schema()
    assert schema.description == "Check whether a text string contains a specific substring."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to check."
    assert inputs["substring"].tooltip == "Substring to search for inside the string."
    assert inputs["case_sensitive"].tooltip == "When enabled, requires character case to match exactly."

    assert StringContains.execute("Hello World", "world", False).args == (True,)
    assert StringContains.execute("Hello World", "world", True).args == (False,)


def test_string_compare_schema_and_execution():
    schema = StringCompare.define_schema()
    assert schema.description == "Compare two text strings using Starts With, Ends With, or Equal comparisons."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string_a"].tooltip == "First text string."
    assert inputs["string_b"].tooltip == "Second text string."
    assert inputs["mode"].tooltip == "Comparison method: Starts With, Ends With, or Equal."
    assert inputs["case_sensitive"].tooltip == "When enabled, matches character case exactly."

    assert StringCompare.execute("ComfyUI", "Comfy", "Starts With", True).args == (True,)
    assert StringCompare.execute("ComfyUI", "ui", "Ends With", False).args == (True,)
    assert StringCompare.execute("ComfyUI", "comfyui", "Equal", False).args == (True,)


def test_regex_match_schema_and_execution():
    schema = RegexMatch.define_schema()
    assert schema.description == "Test whether a text string matches a regular expression pattern."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to test."
    assert inputs["regex_pattern"].tooltip == "Regular expression pattern to search for."
    assert inputs["case_insensitive"].tooltip == "When enabled, ignores letter case during matching."

    assert RegexMatch.execute("Test 123", r"\d+", True, False, False).args == (True,)


def test_regex_extract_schema_and_execution():
    schema = RegexExtract.define_schema()
    assert schema.description == "Extract matching substrings or regex capture groups from a text string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to extract from."
    assert inputs["regex_pattern"].tooltip == "Regular expression pattern with optional capture groups."
    assert inputs["mode"].tooltip == "Extraction strategy: First Match, All Matches, First Group, or All Groups."

    assert RegexExtract.execute("User: Alice, ID: 42", r"ID: (\d+)", "First Group", True, False, False, 1).args == ("42",)


def test_regex_replace_schema_and_execution():
    schema = RegexReplace.define_schema()
    assert schema.description == "Find and replace text using regex patterns."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "Input text string to search within."
    assert inputs["regex_pattern"].tooltip == "Regular expression pattern to find."
    assert inputs["replace"].tooltip == "Replacement string (supports regex backreferences like \\1)."

    assert RegexReplace.execute("abc 123 def 456", r"\d+", "#").args == ("abc # def #",)
