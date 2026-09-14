import sys
import unittest
from unittest.mock import MagicMock

# Mock CUDA / model management dependencies if necessary for unit test environment
sys.modules["comfy.model_management"] = MagicMock()

from comfy_extras.nodes_compositing import PorterDuffImageComposite


class TestPorterDuffImageCompositeUX(unittest.TestCase):
    def test_porter_duff_image_composite_schema(self):
        schema = PorterDuffImageComposite.define_schema()

        # Verify schema node description
        self.assertTrue(hasattr(schema, "description"))
        self.assertIsNotNone(schema.description)
        self.assertIn("Porter-Duff", schema.description)

        # Verify search aliases
        self.assertTrue(hasattr(schema, "search_aliases"))
        self.assertIn("alpha composite", schema.search_aliases)

        # Verify input parameter tooltips
        input_dict = {inp.name: inp for inp in schema.inputs}
        for param_name in ["source", "source_alpha", "destination", "destination_alpha", "mode"]:
            self.assertIn(param_name, input_dict)
            self.assertTrue(hasattr(input_dict[param_name], "tooltip"))
            self.assertIsNotNone(input_dict[param_name].tooltip)
            self.assertGreater(len(input_dict[param_name].tooltip), 0)

        # Verify output parameter tooltips
        for output in schema.outputs:
            self.assertTrue(hasattr(output, "tooltip"))
            self.assertIsNotNone(output.tooltip)
            self.assertGreater(len(output.tooltip), 0)


if __name__ == "__main__":
    unittest.main()
