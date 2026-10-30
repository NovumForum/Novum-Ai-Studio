from __future__ import annotations
from typing import TypedDict
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

# sentinel for missing inputs
MISSING = object()


class SwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySwitchNode",
            display_name="Switch",
            category="logic",
            description="Routes execution between two inputs based on a boolean condition.",
            search_aliases=["switch", "if", "branch", "condition", "toggle", "if else"],
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch", tooltip="When true, outputs on_true; when false, outputs on_false."),
                io.MatchType.Input("on_false", template=template, lazy=True, tooltip="Value or condition to output when switch is false."),
                io.MatchType.Input("on_true", template=template, lazy=True, tooltip="Value or condition to output when switch is true."),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output", tooltip="The selected input value (on_true or on_false) based on switch."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=None, on_true=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, switch, on_true, on_false) -> io.NodeOutput:
        return io.NodeOutput(on_true if switch else on_false)


class SoftSwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySoftSwitchNode",
            display_name="Soft Switch",
            category="logic",
            description="Routes execution between two inputs based on a boolean condition, allowing fallback when an input is disconnected.",
            search_aliases=["soft switch", "conditional fallback", "optional switch"],
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch", tooltip="When true, outputs on_true; when false, outputs on_false."),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True, tooltip="Optional value or condition to output when switch is false."),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True, tooltip="Optional value or condition to output when switch is true."),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output", tooltip="The selected input value based on switch state and input availability."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=MISSING, on_true=MISSING):
        # We use MISSING instead of None, as None is passed for connected-but-unevaluated inputs.
        # This trick allows us to ignore the value of the switch and still be able to run execute().

        # One of the inputs may be missing, in which case we need to evaluate the other input
        if on_false is MISSING:
            return ["on_true"]
        if on_true is MISSING:
            return ["on_false"]
        # Normal lazy switch operation
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def validate_inputs(cls, switch, on_false=MISSING, on_true=MISSING):
        # This check happens before check_lazy_status(), so we can eliminate the case where
        # both inputs are missing.
        if on_false is MISSING and on_true is MISSING:
            return "At least one of on_false or on_true must be connected to Switch node"
        return True

    @classmethod
    def execute(cls, switch, on_true=MISSING, on_false=MISSING) -> io.NodeOutput:
        if on_true is MISSING:
            return io.NodeOutput(on_false)
        if on_false is MISSING:
            return io.NodeOutput(on_true)
        return io.NodeOutput(on_true if switch else on_false)


class CustomComboNode(io.ComfyNode):
    """
    Frontend node that allows user to write their own options for a combo.
    This is here to make sure the node has a backend-representation to avoid some annoyances.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CustomCombo",
            display_name="Custom Combo",
            category="utils",
            description="Allows dynamic user-defined combo dropdown selections.",
            search_aliases=["custom combo", "dynamic dropdown", "user combo"],
            is_experimental=True,
            inputs=[io.Combo.Input("choice", options=[], tooltip="The selected option from the custom combo list.")],
            outputs=[
                io.String.Output(display_name="STRING", tooltip="The selected string option."),
                io.Int.Output(display_name="INDEX", tooltip="The index of the selected option."),
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def validate_inputs(cls, choice: io.Combo.Type, index: int = 0, **kwargs) -> bool:
        # NOTE: DO NOT DO THIS unless you want to skip validation entirely on the node's inputs.
        # I am doing that here because the widgets (besides the combo dropdown) on this node are fully frontend defined.
        # I need to skip checking that the chosen combo option is in the options list, since those are defined by the user.
        return True

    @classmethod
    def execute(cls, choice: io.Combo.Type, index: int = 0, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(choice, index)


class DCTestNode(io.ComfyNode):
    class DCValues(TypedDict):
        combo: str
        string: str
        integer: int
        image: io.Image.Type
        subcombo: dict[str]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DCTestNode",
            display_name="DCTest",
            category="logic",
            is_output_node=True,
            inputs=[io.DynamicCombo.Input("combo", options=[
                io.DynamicCombo.Option("option1", [io.String.Input("string")]),
                io.DynamicCombo.Option("option2", [io.Int.Input("integer")]),
                io.DynamicCombo.Option("option3", [io.Image.Input("image")]),
                io.DynamicCombo.Option("option4", [
                    io.DynamicCombo.Input("subcombo", options=[
                        io.DynamicCombo.Option("opt1", [io.Float.Input("float_x"), io.Float.Input("float_y")]),
                        io.DynamicCombo.Option("opt2", [io.Mask.Input("mask1", optional=True)]),
                    ])
                ])]
            )],
            outputs=[io.AnyType.Output()],
        )

    @classmethod
    def execute(cls, combo: DCValues) -> io.NodeOutput:
        combo_val = combo["combo"]
        if combo_val == "option1":
            return io.NodeOutput(combo["string"])
        elif combo_val == "option2":
            return io.NodeOutput(combo["integer"])
        elif combo_val == "option3":
            return io.NodeOutput(combo["image"])
        elif combo_val == "option4":
            return io.NodeOutput(f"{combo['subcombo']}")
        else:
            raise ValueError(f"Invalid combo: {combo_val}")


class AutogrowNamesTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplateNames(input=io.Float.Input("float"), names=["a", "b", "c"])
        return io.Schema(
            node_id="AutogrowNamesTestNode",
            display_name="AutogrowNamesTest",
            category="logic",
            inputs=[
                _io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ",".join([str(x) for x in vals])
        return io.NodeOutput(combined)

class AutogrowPrefixTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplatePrefix(input=io.Float.Input("float"), prefix="float", min=1, max=10)
        return io.Schema(
            node_id="AutogrowPrefixTestNode",
            display_name="AutogrowPrefixTest",
            category="logic",
            inputs=[
                _io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ",".join([str(x) for x in vals])
        return io.NodeOutput(combined)

class ComboOutputTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComboOptionTestNode",
            display_name="ComboOptionTest",
            category="logic",
            inputs=[io.Combo.Input("combo", options=["option1", "option2", "option3"]),
                    io.Combo.Input("combo2", options=["option4", "option5", "option6"])],
            outputs=[io.Combo.Output(), io.Combo.Output()],
        )

    @classmethod
    def execute(cls, combo: io.Combo.Type, combo2: io.Combo.Type) -> io.NodeOutput:
        return io.NodeOutput(combo, combo2)

class ConvertStringToComboNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertStringToComboNode",
            description="Converts a string input into a combo dropdown selection.",
            search_aliases=["string to dropdown", "text to combo"],
            display_name="Convert String to Combo",
            category="logic",
            inputs=[io.String.Input("string", tooltip="The string value to convert to a combo option.")],
            outputs=[io.Combo.Output(tooltip="The resulting combo option.")],
        )

    @classmethod
    def execute(cls, string: str) -> io.NodeOutput:
        return io.NodeOutput(string)

class InvertBooleanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InvertBooleanNode",
            description="Inverts a boolean value (True becomes False, False becomes True).",
            search_aliases=["not", "toggle", "negate", "flip boolean"],
            display_name="Invert Boolean",
            category="logic",
            inputs=[io.Boolean.Input("boolean", tooltip="The boolean value to invert.")],
            outputs=[io.Boolean.Output(tooltip="The inverted boolean value.")],
        )

    @classmethod
    def execute(cls, boolean: bool) -> io.NodeOutput:
        return io.NodeOutput(not boolean)

class LogicExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SwitchNode,
            CustomComboNode,
            # SoftSwitchNode,
            # ConvertStringToComboNode,
            # DCTestNode,
            # AutogrowNamesTestNode,
            # AutogrowPrefixTestNode,
            # ComboOutputTestNode,
            # InvertBooleanNode,
        ]

async def comfy_entrypoint() -> LogicExtension:
    return LogicExtension()
