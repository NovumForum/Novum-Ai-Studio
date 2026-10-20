import re
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class StringConcatenate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringConcatenate",
            display_name="Concatenate",
            category="utils/string",
            description="Concatenates two text strings together with an optional delimiter.",
            search_aliases=["text concat", "join text", "merge text", "combine strings", "concat", "concatenate", "append text", "combine text", "string"],
            inputs=[
                io.String.Input("string_a", multiline=True, tooltip="First text string to concatenate."),
                io.String.Input("string_b", multiline=True, tooltip="Second text string to concatenate."),
                io.String.Input("delimiter", multiline=False, default="", tooltip="Optional separator inserted between string_a and string_b."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string_a, string_b, delimiter):
        return io.NodeOutput(delimiter.join((string_a, string_b)))


class StringSubstring(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringSubstring",
            display_name="Substring",
            category="utils/string",
            description="Extracts a slice/portion of a text string using start and end indices.",
            search_aliases=["extract text", "text portion", "slice string", "sub text", "string index"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string from which to extract a substring."),
                io.Int.Input("start", tooltip="Zero-based starting index for the slice."),
                io.Int.Input("end", tooltip="Ending index (exclusive) for the slice."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, start, end):
        return io.NodeOutput(string[start:end])


class StringLength(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringLength",
            display_name="Length",
            category="utils/string",
            description="Calculates the total character count of a text string.",
            search_aliases=["character count", "text size", "string length", "len", "count chars"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string whose length will be counted."),
            ],
            outputs=[
                io.Int.Output(display_name="length"),
            ]
        )

    @classmethod
    def execute(cls, string):
        return io.NodeOutput(len(string))


class CaseConverter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CaseConverter",
            display_name="Case Converter",
            category="utils/string",
            description="Converts text casing to uppercase, lowercase, capitalize, or title case.",
            search_aliases=["text case", "uppercase", "lowercase", "capitalize", "title case", "change case"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to change casing."),
                io.Combo.Input("mode", options=["UPPERCASE", "lowercase", "Capitalize", "Title Case"], tooltip="Casing conversion mode to apply."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, mode):
        if mode == "UPPERCASE":
            result = string.upper()
        elif mode == "lowercase":
            result = string.lower()
        elif mode == "Capitalize":
            result = string.capitalize()
        elif mode == "Title Case":
            result = string.title()
        else:
            result = string

        return io.NodeOutput(result)


class StringTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringTrim",
            display_name="Trim",
            category="utils/string",
            description="Trims leading and/or trailing whitespace characters from a text string.",
            search_aliases=["clean whitespace", "remove whitespace", "strip text", "trim spaces"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to trim."),
                io.Combo.Input("mode", options=["Both", "Left", "Right"], tooltip="Where to trim whitespace: both sides, left (leading), or right (trailing)."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, mode):
        if mode == "Both":
            result = string.strip()
        elif mode == "Left":
            result = string.lstrip()
        elif mode == "Right":
            result = string.rstrip()
        else:
            result = string

        return io.NodeOutput(result)


class StringReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringReplace",
            display_name="Replace",
            category="utils/string",
            description="Replaces occurrences of a specified substring with new replacement text.",
            search_aliases=["find and replace", "substitute", "swap text", "replace text", "search replace"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to process."),
                io.String.Input("find", multiline=True, tooltip="Target text/substring to search for and replace."),
                io.String.Input("replace", multiline=True, tooltip="New text to substitute in place of matching target text."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, find, replace):
        return io.NodeOutput(string.replace(find, replace))


class StringContains(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringContains",
            display_name="Contains",
            category="utils/string",
            description="Checks whether a text string contains a specific target substring.",
            search_aliases=["text includes", "string includes", "contains text", "find substring"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to search within."),
                io.String.Input("substring", multiline=True, tooltip="Substring to search for in the input text."),
                io.Boolean.Input("case_sensitive", default=True, advanced=True, tooltip="When enabled, search matches exact letter casing."),
            ],
            outputs=[
                io.Boolean.Output(display_name="contains"),
            ]
        )

    @classmethod
    def execute(cls, string, substring, case_sensitive):
        if case_sensitive:
            contains = substring in string
        else:
            contains = substring.lower() in string.lower()

        return io.NodeOutput(contains)


class StringCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringCompare",
            display_name="Compare",
            category="utils/string",
            description="Compares two text strings for equality, prefix, or suffix match.",
            search_aliases=["text match", "string equals", "starts with", "ends with", "compare text"],
            inputs=[
                io.String.Input("string_a", multiline=True, tooltip="First text string to compare."),
                io.String.Input("string_b", multiline=True, tooltip="Second text string to compare against string_a."),
                io.Combo.Input("mode", options=["Starts With", "Ends With", "Equal"], tooltip="Comparison mode (Equal, Starts With, or Ends With)."),
                io.Boolean.Input("case_sensitive", default=True, advanced=True, tooltip="When enabled, comparison checks exact character casing."),
            ],
            outputs=[
                io.Boolean.Output(),
            ]
        )

    @classmethod
    def execute(cls, string_a, string_b, mode, case_sensitive):
        if case_sensitive:
            a = string_a
            b = string_b
        else:
            a = string_a.lower()
            b = string_b.lower()

        if mode == "Equal":
            return io.NodeOutput(a == b)
        elif mode == "Starts With":
            return io.NodeOutput(a.startswith(b))
        elif mode == "Ends With":
            return io.NodeOutput(a.endswith(b))


class RegexMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexMatch",
            display_name="Regex Match",
            category="utils/string",
            description="Tests if a regular expression pattern matches anywhere within a text string.",
            search_aliases=["pattern match", "text contains", "string match", "regex test", "pattern test"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to test with regex pattern."),
                io.String.Input("regex_pattern", multiline=True, tooltip="Regular expression pattern to match."),
                io.Boolean.Input("case_insensitive", default=True, advanced=True, tooltip="When enabled, matching ignores upper/lower case differences."),
                io.Boolean.Input("multiline", default=False, advanced=True, tooltip="When enabled, ^ and $ match line start/end boundaries."),
                io.Boolean.Input("dotall", default=False, advanced=True, tooltip="When enabled, the dot (.) character matches newlines as well."),
            ],
            outputs=[
                io.Boolean.Output(display_name="matches"),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, case_insensitive, multiline, dotall):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            match = re.search(regex_pattern, string, flags)
            result = match is not None

        except re.error:
            result = False

        return io.NodeOutput(result)


class RegexExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexExtract",
            display_name="Regex Extract",
            category="utils/string",
            description="Extracts matching text or capture groups from a string using regex patterns.",
            search_aliases=["pattern extract", "text parser", "parse text", "regex capture", "extract group"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string from which to extract matches."),
                io.String.Input("regex_pattern", multiline=True, tooltip="Regular expression pattern with optional capture groups."),
                io.Combo.Input("mode", options=["First Match", "All Matches", "First Group", "All Groups"], tooltip="Extraction mode determining which matches or groups to return."),
                io.Boolean.Input("case_insensitive", default=True, advanced=True, tooltip="When enabled, pattern matching ignores case."),
                io.Boolean.Input("multiline", default=False, advanced=True, tooltip="When enabled, ^ and $ match line start/end boundaries."),
                io.Boolean.Input("dotall", default=False, advanced=True, tooltip="When enabled, the dot (.) character matches newline characters."),
                io.Int.Input("group_index", default=1, min=0, max=100, advanced=True, tooltip="Index of the capture group to extract when using group modes."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, mode, case_insensitive, multiline, dotall, group_index):
        join_delimiter = "\n"

        flags = 0
        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            if mode == "First Match":
                match = re.search(regex_pattern, string, flags)
                if match:
                    result = match.group(0)
                else:
                    result = ""

            elif mode == "All Matches":
                matches = re.findall(regex_pattern, string, flags)
                if matches:
                    if isinstance(matches[0], tuple):
                        result = join_delimiter.join([m[0] for m in matches])
                    else:
                        result = join_delimiter.join(matches)
                else:
                    result = ""

            elif mode == "First Group":
                match = re.search(regex_pattern, string, flags)
                if match and len(match.groups()) >= group_index:
                    result = match.group(group_index)
                else:
                    result = ""

            elif mode == "All Groups":
                matches = re.finditer(regex_pattern, string, flags)
                results = []
                for match in matches:
                    if match.groups() and len(match.groups()) >= group_index:
                        results.append(match.group(group_index))
                result = join_delimiter.join(results)
            else:
                result = ""

        except re.error:
            result = ""

        return io.NodeOutput(result)


class RegexReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexReplace",
            display_name="Regex Replace",
            category="utils/string",
            description="Find and replace text using regular expression patterns.",
            search_aliases=["pattern replace", "find and replace", "substitution", "regex replace", "pattern substitute"],
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to perform replacements on."),
                io.String.Input("regex_pattern", multiline=True, tooltip="Regular expression pattern to search for."),
                io.String.Input("replace", multiline=True, tooltip="Replacement text or pattern substitution string."),
                io.Boolean.Input("case_insensitive", default=True, optional=True, advanced=True, tooltip="When enabled, pattern search ignores character casing."),
                io.Boolean.Input("multiline", default=False, optional=True, advanced=True, tooltip="When enabled, ^ and $ match beginning/end of each line."),
                io.Boolean.Input("dotall", default=False, optional=True, advanced=True, tooltip="When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines."),
                io.Int.Input("count", default=0, min=0, max=100, optional=True, advanced=True, tooltip="Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, replace, case_insensitive=True, multiline=False, dotall=False, count=0):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        result = re.sub(regex_pattern, replace, string, count=count, flags=flags)
        return io.NodeOutput(result)


class StringExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
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
        ]

async def comfy_entrypoint() -> StringExtension:
    return StringExtension()
