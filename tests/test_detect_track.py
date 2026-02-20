from src.config.schema import TrackerConfig
from src.pipeline.detect_track import BBox, Detection, TrackManager, bbox_iou


def test_bbox_iou_basic() -> None:
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    iou = bbox_iou(a, b)
    assert 0.1 < iou < 0.2


def test_tracker_persists_track_id() -> None:
    tm = TrackManager(TrackerConfig(iou_threshold=0.2, max_missed=5, min_hits=1))

    f0 = [Detection(BBox(10, 10, 100, 180), 0.9)]
    ids0 = tm.update(f0, frame_index=0)

    f1 = [Detection(BBox(12, 12, 102, 182), 0.88)]
    ids1 = tm.update(f1, frame_index=1)

    assert len(ids0) == 1
    assert len(ids1) == 1
    assert ids0[0] == ids1[0]
