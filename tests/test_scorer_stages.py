"""Optional stage plan for the scorer (`stages=`): an experiment variant can move the label-count thresholds.

Calls without `stages=` must behave exactly as before (tests/test_scorer.py covers that path unchanged).
"""

from scorer import STAGES, prequential, stage_for, train
from tests.simulator import featurized, simulate

EARLY_FULL = [(20, "full"), (0, "defaults")]  # skip temperature and core: fit every feature from 20 labels


def test_stage_for_follows_a_custom_plan():
    assert [stage_for(n, stages=EARLY_FULL) for n in (0, 19, 20, 200)] == ["defaults", "defaults", "full", "full"]


def test_stage_for_accepts_a_plan_in_any_order():
    ascending = [(0, "defaults"), (5, "temperature"), (15, "core")]

    assert [stage_for(n, stages=ascending) for n in (4, 5, 14, 15)] == ["defaults", "temperature", "temperature",
                                                                         "core"]


def test_train_uses_the_custom_plan():
    clips = featurized(simulate(11, 20, muse=2, closed=2, gemini=2))

    assert train(clips, gate=False).stage == "temperature"  # default plan
    assert train(clips, gate=False, stages=EARLY_FULL).stage == "full"


def test_prequential_uses_the_custom_plan_for_every_clip():
    clips = featurized(simulate(12, 22, muse=2, closed=2, gemini=2))

    history = prequential(clips, stages=EARLY_FULL)

    assert [h["stage"] for h in history] == ["defaults"] * 20 + ["full"] * 2


def test_default_plan_is_unchanged():
    assert STAGES == [(60, "full"), (30, "core"), (10, "temperature"), (0, "defaults")]
