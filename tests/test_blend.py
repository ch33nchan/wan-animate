import numpy as np

from src.pipeline.blend import feather_mask, temporal_smooth_frame


def test_temporal_smooth_frame_changes_pixels() -> None:
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = temporal_smooth_frame(a, b, alpha=0.5)
    assert int(out[0, 0, 0]) == 100


def test_feather_mask_shape_preserved() -> None:
    m = np.zeros((16, 16), dtype=np.uint8)
    m[4:12, 4:12] = 255
    out = feather_mask(m, kernel_size=7)
    assert out.shape == m.shape
