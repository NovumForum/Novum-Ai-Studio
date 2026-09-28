import ast
import json


def parse_preview_any_module():
    with open("comfy_extras/nodes_preview_any.py", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    class_def = None
    node_class_mappings = None
    node_display_name_mappings = None

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == "PreviewAny":
            class_def = stmt
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS":
                    node_class_mappings = stmt.value
                elif isinstance(target, ast.Name) and target.id == "NODE_DISPLAY_NAME_MAPPINGS":
                    node_display_name_mappings = stmt.value

    return class_def, node_class_mappings, node_display_name_mappings


def test_preview_any_schema_metadata():
    class_def, node_class_mappings, node_display_name_mappings = parse_preview_any_module()
    assert class_def is not None

    desc = None
    search_aliases = None
    input_tooltips = {}

    for item in class_def.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "DESCRIPTION":
                    desc = item.value.value if isinstance(item.value, ast.Constant) else None
                if isinstance(target, ast.Name) and target.id == "SEARCH_ALIASES":
                    search_aliases = [elt.value for elt in item.value.elts] if isinstance(item.value, ast.List) else None
        elif isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
            for stmt in item.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                    for k, v in zip(stmt.value.keys, stmt.value.values):
                        if isinstance(v, ast.Dict):
                            for sub_k, sub_v in zip(v.keys, v.values):
                                if isinstance(sub_k, ast.Constant) and isinstance(sub_v, ast.Tuple):
                                    for tuple_elt in sub_v.elts:
                                        if isinstance(tuple_elt, ast.Dict):
                                            for dk, dv in zip(tuple_elt.keys, tuple_elt.values):
                                                if isinstance(dk, ast.Constant) and dk.value == "tooltip":
                                                    input_tooltips[sub_k.value] = dv.value

    assert desc == "Displays the input value as formatted text or JSON string in the UI output area. Useful for inspecting intermediate node outputs, debugging, and printing values."
    assert search_aliases == ["show output", "inspect", "debug", "print value", "show text"]
    assert input_tooltips.get("source") == "The input value to preview as text or JSON."


def test_preview_any_execution_logic():
    with open("comfy_extras/nodes_preview_any.py", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    # Locate PreviewAny class node
    class_def = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PreviewAny")
    main_fn = next(node for node in class_def.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    # Compile and evaluate function body dynamically in minimal scope
    fn_code = compile(ast.Module(body=[main_fn], type_ignores=[]), filename="<ast>", mode="exec")
    scope = {"json": json, "isinstance": isinstance, "str": str, "int": int, "float": float, "bool": bool, "Exception": Exception}
    exec(fn_code, scope)  # pylint: disable=exec-used # noqa: S102
    main_func = scope["main"]

    class DummySelf:
        pass

    self_obj = DummySelf()

    assert main_func(self_obj, "Hello World") == {"ui": {"text": ("Hello World",)}}
    assert main_func(self_obj, 123) == {"ui": {"text": ("123",)}}
    assert main_func(self_obj, {"key": "value"}) == {"ui": {"text": (json.dumps({"key": "value"}, indent=4),)}}
    assert main_func(self_obj, None) == {"ui": {"text": ("None",)}}
