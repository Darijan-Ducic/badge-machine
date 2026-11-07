"""tools/reread.py: scaled crop boxes and the Muse pick used to score a paid re-read."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("reread_tool", ROOT / "tools" / "reread.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_scaling_keeps_each_box_centre_and_time_and_scales_its_size():
    assert tool.scale_boxes([[40, 100, 200, 60, 30]], 2.0) == [[40, 70.0, 185.0, 120.0, 60.0]]


def test_scale_one_returns_the_same_boxes():
    boxes = [[0, 10, 20, 30, 40], [33, 11, 21, 31, 41]]
    assert tool.scale_boxes(boxes, 1.0) == boxes


def _sample(*hyps):
    return {"output": {"hypotheses": [{"text": t, "s": s} for t, s in hyps]}}


def test_muse_pick_is_the_most_common_top_hypothesis_over_valid_samples():
    samples = [_sample(("we need more", 0.6), ("we need", 0.3)), _sample(("We need more.", 0.5)),
               _sample(("call mom", 0.9)), {"output": None, "error": "timeout"}]
    assert tool.muse_pick(samples) == "we need more"


def test_muse_pick_without_valid_samples_is_none():
    assert tool.muse_pick([{"output": None, "error": "x"}]) is None
