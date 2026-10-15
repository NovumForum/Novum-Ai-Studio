import ast
import unittest

class TestUpscaleModelNodeTooltips(unittest.TestCase):
    def test_upscale_model_nodes_schema_metadata(self):
        with open("comfy_extras/nodes_upscale_model.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

        # Check UpscaleModelLoader
        self.assertIn("UpscaleModelLoader", classes)
        loader_cls = classes["UpscaleModelLoader"]
        loader_source = ast.unparse(loader_cls)
        self.assertIn("description='Loads a model used to upscale images", loader_source)
        self.assertIn("search_aliases=['load upscale model'", loader_source)
        self.assertIn("tooltip='The upscale model file to load.'", loader_source)
        self.assertIn("tooltip='The loaded upscale model.'", loader_source)

        # Check ImageUpscaleWithModel
        self.assertIn("ImageUpscaleWithModel", classes)
        upscale_cls = classes["ImageUpscaleWithModel"]
        upscale_source = ast.unparse(upscale_cls)
        self.assertIn("description='Upscales an image using a loaded super-resolution upscale model.'", upscale_source)
        self.assertIn("search_aliases=['upscale'", upscale_source)
        self.assertIn("tooltip='The upscale model to perform the image scaling.'", upscale_source)
        self.assertIn("tooltip='The input image to upscale.'", upscale_source)
        self.assertIn("tooltip='The upscaled image.'", upscale_source)

if __name__ == "__main__":
    unittest.main()
