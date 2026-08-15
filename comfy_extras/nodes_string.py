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
            description="Combine two text strings into one with an optional delimiter.",
            search_aliases=["text concat", "join text", "merge text", "combine strings", "concat", "concatenate", "append text", "combine text", "string"],
            inputs=[
                io.String.Input("string_a", multiline=True, tooltip="First text string."),
                io.String.Input("string_b", multiline=True, tooltip="Second text string."),
                io.String.Input("delimiter", multiline=False, default="", tooltip="Optional delimiter inserted between the two strings."),
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
            search_aliases=["extract text", "text portion"],
            display_name="Substring",
            category="utils/string",
            description="Extract a slice of a text string using start and end character indices.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to slice."),
                io.Int.Input("start", tooltip="Starting character index (0-based)."),
                io.Int.Input("end", tooltip="Ending character index (exclusive)."),
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
            search_aliases=["character count", "text size"],
            display_name="Length",
            category="utils/string",
            description="Count the total number of characters in a text string.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to measure."),
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
            search_aliases=["text case", "uppercase", "lowercase", "capitalize"],
            display_name="Case Converter",
            category="utils/string",
            description="Convert text string casing mode (e.g. UPPERCASE, lowercase, Capitalize, Title Case).",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to convert."),
                io.Combo.Input("mode", options=["UPPERCASE", "lowercase", "Capitalize", "Title Case"], tooltip="Casing transformation mode to apply."),
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
            search_aliases=["clean whitespace", "remove whitespace"],
            display_name="Trim",
            category="utils/string",
            description="Strip leading and/or trailing whitespace from a text string.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to trim."),
                io.Combo.Input("mode", options=["Both", "Left", "Right"], tooltip="Trim mode specifying which sides to strip whitespace from."),
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
            search_aliases=["find and replace", "substitute", "swap text"],
            display_name="Replace",
            category="utils/string",
            description="Search for occurrences of a substring and replace them with new text.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to perform replacement on."),
                io.String.Input("find", multiline=True, tooltip="Substring to search for."),
                io.String.Input("replace", multiline=True, tooltip="Text string to replace found occurrences with."),
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
            search_aliases=["text includes", "string includes"],
            display_name="Contains",
            category="utils/string",
            description="Check whether a text string contains a specific substring.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to check."),
                io.String.Input("substring", multiline=True, tooltip="Substring to search for inside the string."),
                io.Boolean.Input("case_sensitive", default=True, advanced=True, tooltip="When enabled, requires character case to match exactly."),
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
            search_aliases=["text match", "string equals", "starts with", "ends with"],
            display_name="Compare",
            category="utils/string",
            description="Compare two text strings using Starts With, Ends With, or Equal comparisons.",
            inputs=[
                io.String.Input("string_a", multiline=True, tooltip="First text string."),
                io.String.Input("string_b", multiline=True, tooltip="Second text string."),
                io.Combo.Input("mode", options=["Starts With", "Ends With", "Equal"], tooltip="Comparison method: Starts With, Ends With, or Equal."),
                io.Boolean.Input("case_sensitive", default=True, advanced=True, tooltip="When enabled, matches character case exactly."),
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
            search_aliases=["pattern match", "text contains", "string match"],
            display_name="Regex Match",
            category="utils/string",
            description="Test whether a text string matches a regular expression pattern.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to test."),
                io.String.Input("regex_pattern", multiline=True, tooltip="Regular expression pattern to search for."),
                io.Boolean.Input("case_insensitive", default=True, advanced=True, tooltip="When enabled, ignores letter case during matching."),
                io.Boolean.Input("multiline", default=False, advanced=True, tooltip="When enabled, ^ and $ match line beginnings and endings."),
                io.Boolean.Input("dotall", default=False, advanced=True, tooltip="When enabled, dot (.) matches newline characters as well."),
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
            search_aliases=["pattern extract", "text parser", "parse text"],
            display_name="Regex Extract",
            category="utils/string",
            description="Extract matching substrings or regex capture groups from a text string.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to extract from."),
                io.String.Input("regex_pattern", multiline=True, tooltip="Regular expression pattern with optional capture groups."),
                io.Combo.Input("mode", options=["First Match", "All Matches", "First Group", "All Groups"], tooltip="Extraction strategy: First Match, All Matches, First Group, or All Groups."),
                io.Boolean.Input("case_insensitive", default=True, advanced=True, tooltip="When enabled, ignores letter case."),
                io.Boolean.Input("multiline", default=False, advanced=True, tooltip="When enabled, ^ and $ match line beginnings and endings."),
                io.Boolean.Input("dotall", default=False, advanced=True, tooltip="When enabled, dot (.) matches newline characters."),
                io.Int.Input("group_index", default=1, min=0, max=100, advanced=True, tooltip="Capture group index (1-based) to extract when using group modes."),
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
            search_aliases=["pattern replace", "find and replace", "substitution"],
            display_name="Regex Replace",
            category="utils/string",
            description="Find and replace text using regex patterns.",
            inputs=[
                io.String.Input("string", multiline=True, tooltip="Input text string to search within."),
                io.String.Input("regex_pattern", multiline=True, tooltip="Regular expression pattern to find."),
                io.String.Input("replace", multiline=True, tooltip="Replacement string (supports regex backreferences like \\1)."),
                io.Boolean.Input("case_insensitive", default=True, optional=True, advanced=True, tooltip="When enabled, ignores letter case during search."),
                io.Boolean.Input("multiline", default=False, optional=True, advanced=True, tooltip="When enabled, ^ and $ match line beginnings and endings."),
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
