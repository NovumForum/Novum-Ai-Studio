import json
from comfy.comfy_types.node_typing import IO

# Preview Any - original implement from
# https://github.com/rgthree/rgthree-comfy/blob/main/py/display_any.py
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class PreviewAny():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"source": (IO.ANY, {"tooltip": "The input value to preview as text or JSON."})},
        }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True

    CATEGORY = "utils"
    DESCRIPTION = "Displays the input value as formatted text or JSON string in the UI output area. Useful for inspecting intermediate node outputs, debugging, and printing values."
    SEARCH_ALIASES = ["show output", "inspect", "debug", "print value", "show text"]

    def main(self, source=None):
        value = 'None'
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source, indent=4)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'source exists, but could not be serialized.'

        return {"ui": {"text": (value,)}}

NODE_CLASS_MAPPINGS = {
    "PreviewAny": PreviewAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewAny": "Preview as Text",
}
