"""Features from open outputs of a schema variant (muse_sampler.SCHEMAS, schema_version != "v1") that drops fields:
each dropped field's feature gets its stated replacement, and v1 outputs keep the documented v1 readings.

Fake visemes (tests/simulator.py): each letter is a code, repeats collapsed; distance = Levenshtein / longer length.
  "go on" -> g o n     "go" -> g o     "stop" -> s t o p     "help me" -> h e l p m e     "wait" -> w a i t
"""

import pytest

from candidates import FEATURES, featurize
from tests.simulator import fake_to_visemes, fake_viseme_distance

F = {name: i for i, name in enumerate(FEATURES)}
BANK = {"phrases": ["stop", "help me", "wait", "go"], "counts": {}}


def h1(*pairs):
    return {"output": {"schema_version": "h1", "hypotheses": [{"text": t, "s": s} for t, s in pairs]}}


def run(samples, bank=BANK):
    return featurize(samples, None, None, bank, to_visemes=fake_to_visemes, viseme_distance=fake_viseme_distance)


def row(clip, text):
    return clip.features[clip.pool.index(text)]


def test_a_variant_without_quality_is_not_counted_as_bad_quality():
    clip = run([h1(("go on", 0.6))])
    assert clip.features[-1, F["bad_quality"]] == 0.0


def test_a_v1_output_without_quality_still_counts_as_bad_quality():
    v1 = {"output": {"schema_version": "v1", "hypotheses": [{"text": "go on", "s": 0.6}]}}
    assert run([v1]).features[-1, F["bad_quality"]] == 1.0


def test_a_variant_sample_with_no_hypotheses_counts_as_no_speech():
    clip = run([h1(("go on", 0.6)), h1(("no", 0.3)), h1()])  # one of three samples saw no speech
    assert clip.features[-1, F["no_speech"]] == pytest.approx(1 / 3)


def test_without_visemes_a_variant_observes_its_top_hypothesis_visemes():
    clip = run([h1(("no", 0.2), ("go on", 0.6))])  # top by s is "go on" -> g o n
    assert row(clip, "go on")[F["viseme_sim"]] == 1.0  # distance 0 to its own visemes
    assert row(clip, "no")[F["viseme_sim"]] == pytest.approx(1 / 3)  # n o vs g o n: 2 edits / 3


def test_without_visemes_the_bank_entries_nearest_the_top_hypothesis_join_the_pool():
    # "go" (g o) is 1 edit from g o n; stop / help me / wait are farther. Without observed visemes every entry would
    # tie and the first three in bank order (stop, help me, wait) would be taken instead.
    clip = run([h1(("go on", 0.6))])
    assert "go" in clip.pool and "wait" not in clip.pool


def test_without_slots_a_variant_supports_words_from_all_its_hypotheses():
    clip = run([h1(("go on", 0.6), ("no", 0.3))], bank={"phrases": ["go up"], "counts": {}})
    assert row(clip, "go up")[F["slot_support"]] == 0.5  # "go" is among go/on/no, "up" is not


def test_a_variant_that_keeps_visemes_uses_them():
    vh1 = {"output": {"schema_version": "vh1", "visemes": [{"v": "s", "f": [0, 1]}, {"v": "t", "f": [2, 3]},
                                                           {"v": "o", "f": [4, 5]}, {"v": "p", "f": [6, 7]}],
                      "hypotheses": [{"text": "go on", "s": 0.6}]}}
    clip = run([vh1])
    assert row(clip, "stop")[F["viseme_sim"]] == 1.0  # observed s t o p, not the hypothesis' g o n
