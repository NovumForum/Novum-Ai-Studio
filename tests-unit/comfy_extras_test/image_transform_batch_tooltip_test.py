import ast
import os

def parse_schema_call(class_node):
    """Helper to parse IO.Schema keyword arguments from a define_schema method."""
    define_schema_fn = None
    for body_item in class_node.body:
        if isinstance(body_item, ast.FunctionDef) and body_item.name == "define_schema":
            define_schema_fn = body_item
            break
    assert define_schema_fn is not None, f"define_schema method missing in {class_node.name}"

    schema_call = None
    for stmt in ast.walk(define_schema_fn):
        if isinstance(stmt, ast.Call):
            if (isinstance(stmt.func, ast.Attribute) and stmt.func.attr == "Schema") or \
               (isinstance(stmt.func, ast.Name) and stmt.func.id == "Schema"):
                schema_call = stmt
                break
    assert schema_call is not None, f"IO.Schema call not found in define_schema of {class_node.name}"

    kwargs = {keyword.arg: keyword.value for keyword in schema_call.keywords}
    return kwargs


class TestImageTransformBatchSchemaMetadata:

    def get_ast_class_map(self):
        filepath = os.path.join("comfy_extras", "nodes_images.py")
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        class_map = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                class_map[stmt.name] = stmt
        return class_map

    def test_repeat_image_batch_metadata(self):
        classes = self.get_ast_class_map()
        assert "RepeatImageBatch" in classes
        kwargs = parse_schema_call(classes["RepeatImageBatch"])

        assert "display_name" in kwargs
        assert ast.literal_eval(kwargs["display_name"]) == "Repeat Image Batch"

        assert "description" in kwargs
        assert "Duplicates an input image" in ast.literal_eval(kwargs["description"])

        assert "search_aliases" in kwargs
        aliases = [ast.literal_eval(elt) for elt in kwargs["search_aliases"].elts]
        assert "repeat batch" in aliases
        assert "copy batch" in aliases

        assert "inputs" in kwargs
        inputs_list = kwargs["inputs"].elts
        assert len(inputs_list) == 2

        # Check input tooltips
        input_tooltips = {}
        for inp in inputs_list:
            inp_kwargs = {kw.arg: kw.value for kw in inp.keywords}
            param_name = ast.literal_eval(inp.args[0])
            assert "tooltip" in inp_kwargs, f"Missing tooltip on input param '{param_name}' in RepeatImageBatch"
            input_tooltips[param_name] = ast.literal_eval(inp_kwargs["tooltip"])

        assert "image" in input_tooltips
        assert "amount" in input_tooltips

        # Check output tooltips
        assert "outputs" in kwargs
        outputs_list = kwargs["outputs"].elts
        out_kwargs = {kw.arg: kw.value for kw in outputs_list[0].keywords}
        assert "tooltip" in out_kwargs
        assert len(ast.literal_eval(out_kwargs["tooltip"])) > 0

    def test_image_from_batch_metadata(self):
        classes = self.get_ast_class_map()
        assert "ImageFromBatch" in classes
        kwargs = parse_schema_call(classes["ImageFromBatch"])

        assert "display_name" in kwargs
        assert ast.literal_eval(kwargs["display_name"]) == "Image From Batch"

        assert "description" in kwargs
        assert "Extracts a sub-batch" in ast.literal_eval(kwargs["description"])

        assert "search_aliases" in kwargs
        aliases = [ast.literal_eval(elt) for elt in kwargs["search_aliases"].elts]
        assert "slice batch" in aliases

        assert "inputs" in kwargs
        inputs_list = kwargs["inputs"].elts
        assert len(inputs_list) == 3

        for inp in inputs_list:
            inp_kwargs = {kw.arg: kw.value for kw in inp.keywords}
            param_name = ast.literal_eval(inp.args[0])
            assert "tooltip" in inp_kwargs, f"Missing tooltip on input param '{param_name}' in ImageFromBatch"

        assert "outputs" in kwargs
        outputs_list = kwargs["outputs"].elts
        out_kwargs = {kw.arg: kw.value for kw in outputs_list[0].keywords}
        assert "tooltip" in out_kwargs

    def test_image_rotate_metadata(self):
        classes = self.get_ast_class_map()
        assert "ImageRotate" in classes
        kwargs = parse_schema_call(classes["ImageRotate"])

        assert "display_name" in kwargs
        assert ast.literal_eval(kwargs["display_name"]) == "Image Rotate"

        assert "description" in kwargs
        assert "Rotates an input image" in ast.literal_eval(kwargs["description"])

        assert "search_aliases" in kwargs
        aliases = [ast.literal_eval(elt) for elt in kwargs["search_aliases"].elts]
        assert "rotate image" in aliases

        assert "inputs" in kwargs
        inputs_list = kwargs["inputs"].elts
        for inp in inputs_list:
            inp_kwargs = {kw.arg: kw.value for kw in inp.keywords}
            param_name = ast.literal_eval(inp.args[0])
            assert "tooltip" in inp_kwargs, f"Missing tooltip on input param '{param_name}' in ImageRotate"

        assert "outputs" in kwargs
        outputs_list = kwargs["outputs"].elts
        out_kwargs = {kw.arg: kw.value for kw in outputs_list[0].keywords}
        assert "tooltip" in out_kwargs

    def test_image_flip_metadata(self):
        classes = self.get_ast_class_map()
        assert "ImageFlip" in classes
        kwargs = parse_schema_call(classes["ImageFlip"])

        assert "display_name" in kwargs
        assert ast.literal_eval(kwargs["display_name"]) == "Image Flip"

        assert "description" in kwargs
        assert "Flips an input image" in ast.literal_eval(kwargs["description"])

        assert "search_aliases" in kwargs
        aliases = [ast.literal_eval(elt) for elt in kwargs["search_aliases"].elts]
        assert "flip image" in aliases

        assert "inputs" in kwargs
        inputs_list = kwargs["inputs"].elts
        for inp in inputs_list:
            inp_kwargs = {kw.arg: kw.value for kw in inp.keywords}
            param_name = ast.literal_eval(inp.args[0])
            assert "tooltip" in inp_kwargs, f"Missing tooltip on input param '{param_name}' in ImageFlip"

        assert "outputs" in kwargs
        outputs_list = kwargs["outputs"].elts
        out_kwargs = {kw.arg: kw.value for kw in outputs_list[0].keywords}
        assert "tooltip" in out_kwargs
