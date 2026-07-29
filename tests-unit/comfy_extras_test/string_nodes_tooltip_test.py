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


def test_string_concatenate_tooltip():
    schema = StringConcatenate.define_schema()
    assert schema.description == "Concatenates two text strings together using an optional delimiter."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string_a"].tooltip == "The first text string."
    assert inputs["string_b"].tooltip == "The second text string to append to the first text string."
    assert inputs["delimiter"].tooltip == "Optional character or string to insert between string_a and string_b (e.g. a space, comma, or newline)."


def test_string_substring_tooltip():
    schema = StringSubstring.define_schema()
    assert schema.description == "Extracts a portion of a string (slice) using starting and ending character indexes."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The source text string to extract from."
    assert inputs["start"].tooltip == "Starting character index (0-based) for the substring slice. Negative values count from the end of the string."
    assert inputs["end"].tooltip == "Ending character index (0-based, exclusive) for the substring slice. Negative values count from the end of the string."


def test_string_length_tooltip():
    schema = StringLength.define_schema()
    assert schema.description == "Calculates the total number of characters (length) in the text string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The text string to measure."


def test_case_converter_tooltip():
    schema = CaseConverter.define_schema()
    assert schema.description == "Converts the character casing of a text string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The input text string to convert."
    assert inputs["mode"].tooltip == "The target casing mode: UPPERCASE, lowercase, Capitalize (first character), or Title Case (first letter of each word)."


def test_string_trim_tooltip():
    schema = StringTrim.define_schema()
    assert schema.description == "Removes leading and/or trailing whitespace (spaces, tabs, newlines) from a text string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The text string to trim."
    assert inputs["mode"].tooltip == "Whether to strip whitespace from both ends, left side only, or right side only."


def test_string_replace_tooltip():
    schema = StringReplace.define_schema()
    assert schema.description == "Replaces all occurrences of a search string with a replacement string."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The source text string to perform replacement on."
    assert inputs["find"].tooltip == "The exact text substring to search for."
    assert inputs["replace"].tooltip == "The replacement text string to substitute in place of found occurrences."


def test_string_contains_tooltip():
    schema = StringContains.define_schema()
    assert schema.description == "Checks if a substring exists within the source string, returning a boolean value."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The source text string to search within."
    assert inputs["substring"].tooltip == "The substring to search for."
    assert inputs["case_sensitive"].tooltip == "When enabled, requires exact case matching. When disabled, searches ignoring character case."


def test_string_compare_tooltip():
    schema = StringCompare.define_schema()
    assert schema.description == "Compares two strings to verify if they are equal, or if one starts/ends with the other."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string_a"].tooltip == "The first text string."
    assert inputs["string_b"].tooltip == "The second text string to compare against the first."
    assert inputs["mode"].tooltip == "The comparison rule: check if string_a starts with, ends with, or is exactly equal to string_b."
    assert inputs["case_sensitive"].tooltip == "When enabled, comparison is case-sensitive."


def test_regex_match_tooltip():
    schema = RegexMatch.define_schema()
    assert schema.description == "Checks if a regular expression pattern matches anywhere inside the string, returning a boolean."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The source text string to evaluate."
    assert inputs["regex_pattern"].tooltip == "The regular expression pattern to match."
    assert inputs["case_insensitive"].tooltip == "When enabled, evaluates ignoring case."
    assert inputs["multiline"].tooltip == "When enabled, '^' and '$' match the start/end of individual lines in multiline strings rather than the start/end of the whole string."
    assert inputs["dotall"].tooltip == "When enabled, the dot (.) matches any character including newline characters."


def test_regex_extract_tooltip():
    schema = RegexExtract.define_schema()
    assert schema.description == "Extracts matches or captured groups from a text string using a regular expression pattern."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The source text string to extract from."
    assert inputs["regex_pattern"].tooltip == "The regular expression pattern with optional capture groups."
    assert inputs["mode"].tooltip == "Extraction mode: First Match (first occurrence), All Matches (joined by newline), First Group (first captured group), or All Groups (all captured groups joined)."
    assert inputs["case_insensitive"].tooltip == "When enabled, matches ignoring case."
    assert inputs["multiline"].tooltip == "When enabled, '^' and '$' match start/end of individual lines."
    assert inputs["dotall"].tooltip == "When enabled, the dot (.) matches any character including newline characters."
    assert inputs["group_index"].tooltip == "The index of the regex capture group to extract (0 for full match, 1 for the first captured group, etc.). Only used in Group modes."


def test_regex_replace_tooltip():
    schema = RegexReplace.define_schema()
    assert schema.description == "Find and replace text using regex patterns."
    inputs = {inp.id: inp for inp in schema.inputs}
    assert inputs["string"].tooltip == "The source text string to perform replacement on."
    assert inputs["regex_pattern"].tooltip == "The regular expression search pattern."
    assert inputs["replace"].tooltip == "The replacement text string (supports backreferences like \\1)."
    assert inputs["case_insensitive"].tooltip == "When enabled, matches ignoring case."
    assert inputs["multiline"].tooltip == "When enabled, '^' and '$' match the start/end of individual lines in multiline strings."
    assert inputs["dotall"].tooltip == "When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines."
    assert inputs["count"].tooltip == "Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc."
