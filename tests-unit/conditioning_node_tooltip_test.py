import ast
from pathlib import Path


def get_class_node_from_ast(tree: ast.AST, class_name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"Class '{class_name}' not found in AST")


def test_conditioning_nodes_schema_metadata():
    file_path = Path("nodes.py")
    tree = ast.parse(file_path.read_text(encoding="utf-8"))

    nodes_to_check = {
        "ConditioningCombine": {
            "search_aliases": ["combine", "merge conditioning"],
            "inputs": ["conditioning_1", "conditioning_2"],
            "outputs_len": 1,
        },
        "ConditioningAverage": {
            "search_aliases": ["blend prompts", "interpolate conditioning"],
            "inputs": ["conditioning_to", "conditioning_from", "conditioning_to_strength"],
            "outputs_len": 1,
        },
        "ConditioningConcat": {
            "search_aliases": ["concat conditioning", "append conditioning"],
            "inputs": ["conditioning_to", "conditioning_from"],
            "outputs_len": 1,
        },
        "ConditioningSetAreaPercentage": {
            "search_aliases": ["regional prompt percentage", "area prompt percent"],
            "inputs": ["conditioning", "width", "height", "x", "y", "strength"],
            "outputs_len": 1,
        },
        "ConditioningSetAreaStrength": {
            "search_aliases": ["area strength", "regional prompt strength"],
            "inputs": ["conditioning", "strength"],
            "outputs_len": 1,
        },
    }

    for class_name, expected in nodes_to_check.items():
        class_ast = get_class_node_from_ast(tree, class_name)

        # Check DESCRIPTION attribute
        desc_assign = next(
            (
                stmt
                for stmt in class_ast.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "DESCRIPTION"
                    for target in stmt.targets
                )
            ),
            None,
        )
        assert desc_assign is not None, f"{class_name} missing DESCRIPTION attribute"
        assert isinstance(desc_assign.value, ast.Constant) and isinstance(desc_assign.value.value, str)
        assert len(desc_assign.value.value) > 10, f"{class_name} DESCRIPTION is too short"

        # Check SEARCH_ALIASES attribute
        aliases_assign = next(
            (
                stmt
                for stmt in class_ast.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "SEARCH_ALIASES"
                    for target in stmt.targets
                )
            ),
            None,
        )
        assert aliases_assign is not None, f"{class_name} missing SEARCH_ALIASES attribute"
        assert isinstance(aliases_assign.value, ast.List)
        aliases = [e.value for e in aliases_assign.value.elts if isinstance(e, ast.Constant)]
        for expected_alias in expected["search_aliases"]:
            assert expected_alias in aliases, f"{expected_alias} not in SEARCH_ALIASES of {class_name}"

        # Check OUTPUT_TOOLTIPS attribute
        out_tooltips_assign = next(
            (
                stmt
                for stmt in class_ast.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "OUTPUT_TOOLTIPS"
                    for target in stmt.targets
                )
            ),
            None,
        )
        assert out_tooltips_assign is not None, f"{class_name} missing OUTPUT_TOOLTIPS attribute"
        assert isinstance(out_tooltips_assign.value, ast.Tuple)
        assert len(out_tooltips_assign.value.elts) == expected["outputs_len"]

        # Check INPUT_TYPES function and tooltips
        input_types_fn = next(
            (
                stmt
                for stmt in class_ast.body
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "INPUT_TYPES"
            ),
            None,
        )
        assert input_types_fn is not None, f"{class_name} missing INPUT_TYPES method"

        # Check parameter tooltips in INPUT_TYPES AST
        code_str = ast.unparse(input_types_fn)
        for param in expected["inputs"]:
            assert f"'{param}'" in code_str or f'"{param}"' in code_str, f"Parameter {param} missing in {class_name}"
            assert "tooltip" in code_str, f"Tooltip missing in {class_name} INPUT_TYPES"
