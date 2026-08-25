import math
import functools
from comfy_extras.nodes_post_processing import normalized_bayer_matrix


def test_normalized_bayer_matrix_cache_and_correctness():
    # Test orders 2, 4, 8, 16
    for order in [2, 4, 8, 16]:
        bayer_n = int(math.log2(order))
        m = normalized_bayer_matrix(bayer_n)
        assert m.shape == (order, order)

    # Verify that lru_cache hits increase on repeated calls
    cache_info_before = normalized_bayer_matrix.cache_info()
    m_repeat = normalized_bayer_matrix(int(math.log2(8)))
    cache_info_after = normalized_bayer_matrix.cache_info()
    assert cache_info_after.hits > cache_info_before.hits
