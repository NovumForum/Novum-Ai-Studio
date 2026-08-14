import pytest
from comfy_extras.nodes_latent import (
    LatentAdd,
    LatentSubtract,
    LatentMultiply,
    LatentInterpolate,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentBatch,
    LatentBatchSeedBehavior,
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationTonemapReinhard,
    LatentOperationSharpen,
    ReplaceVideoLatentFrames,
)

NODES_TO_TEST = [
    LatentAdd,
    LatentSubtract,
    LatentMultiply,
    LatentInterpolate,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentBatch,
    LatentBatchSeedBehavior,
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationTonemapReinhard,
    LatentOperationSharpen,
    ReplaceVideoLatentFrames,
]


@pytest.mark.parametrize("node_cls", NODES_TO_TEST)
def test_latent_node_schema_metadata_and_tooltips(node_cls):
    schema = node_cls.define_schema()

    # Verify node description and search aliases
    assert schema.description is not None and len(schema.description.strip()) > 0, f"{node_cls.__name__} missing description"
    assert schema.search_aliases is not None and len(schema.search_aliases) > 0, f"{node_cls.__name__} missing search_aliases"

    # Verify input tooltips
    for inp in schema.inputs:
        assert inp.tooltip is not None and len(inp.tooltip.strip()) > 0, f"{node_cls.__name__} input {inp.id} missing tooltip"

    # Verify output tooltips
    for out in schema.outputs:
        assert out.tooltip is not None and len(out.tooltip.strip()) > 0, f"{node_cls.__name__} output {out.id} missing tooltip"
