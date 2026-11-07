"""Evaluator adapter: chronological splits, metric checks, routing/context replay of raw records, train failures."""

import sys
import types

import numpy as np
import pytest

import evaluator
from candidates import FEATURES, featurize
from tests.sim_world import OFF_BANK, world
from versions import default_config

COL = {name: FEATURES.index(name) for name in FEATURES}


def config(**knobs):
    c = default_config()
    c.update(knobs)
    return c


# ---------------------------------------------------------------- splits


def test_splits_are_chronological_60_20_20_over_labeled_clips():
    records = [{"id": i, "stt": "x"} for i in range(10)] + [{"id": "unlabeled", "stt": None}]
    parts = {name: [r["id"] for r in evaluator.split(records, name)] for name in evaluator.SPLITS}
    assert parts == {"train": [0, 1, 2, 3, 4, 5], "heldout": [6, 7], "test": [8, 9]}


def test_unknown_split_is_refused():
    with pytest.raises(ValueError):
        evaluator.split([{"id": 1, "stt": "x"}], "validation")


# ---------------------------------------------------------------- metric checks

GOOD = {"clips": 80, "word_acc": 0.61, "sentence_acc": 0.5, "viseme_acc": 0.7, "top3_acc": 0.8,
        "autosend_precision": None, "autosend_coverage": None, "cost_per_clip": None}


def test_metrics_are_projected_onto_the_contract_keys():
    checked = evaluator.check_metrics({**GOOD, "gemini_acc": 0.3, "pool_recall": 0.9})
    assert set(checked) == set(evaluator.METRICS)


def test_negative_word_accuracy_is_allowed():  # 1 - WER goes below 0 when VTT inserts words
    assert evaluator.check_metrics({**GOOD, "word_acc": -0.2})["word_acc"] == -0.2


@pytest.mark.parametrize("bad", [{"word_acc": 61.0}, {"viseme_acc": 70.0}, {"sentence_acc": 1.5},
                                 {"top3_acc": -0.1}, {"autosend_coverage": 30.0}])
def test_percent_scale_or_out_of_range_metrics_are_refused(bad):
    with pytest.raises(ValueError):
        evaluator.check_metrics({**GOOD, **bad})


def test_missing_metric_is_refused():
    with pytest.raises(ValueError):
        evaluator.check_metrics({k: v for k, v in GOOD.items() if k != "viseme_acc"})


# ---------------------------------------------------------------- routing + context replay


@pytest.fixture(scope="module")
def records():
    return world(n=30, seed=3)


def _featurized(record, **parts):
    args = {k: record[k] for k in ("samples", "closed", "gemini", "bank", "stt")}
    args.update(parts)
    import visemes
    return featurize(**args, to_visemes=visemes.to_visemes, viseme_distance=visemes.viseme_distance)


def test_k_uses_the_first_k_stored_samples(records):
    record = records[0]
    clip = evaluator.replay([record], config(k=2))[0]
    expected = _featurized(record, samples=record["samples"][:2])
    assert clip.pool == expected.pool
    np.testing.assert_allclose(clip.features, expected.features)


def test_sources_drop_the_calls_left_out(records):
    clips = evaluator.replay(records, config(sources=["open"]))
    assert all(np.all(c.features[:, COL["closed"]] == 0) and np.all(c.features[:, COL["gemini"]] == 0) for c in clips)
    full = evaluator.replay(records, config())
    assert any(np.any(c.features[:, COL["gemini"]] > 0) for c in full)


def _agreeing(record, agree: bool):
    samples = []
    for i, s in enumerate(record["samples"][:3]):
        out = dict(s["output"])
        text = "call mom" if agree or i == 0 else ["stop", "yes please"][i - 1]
        out["hypotheses"] = [{"text": text, "s": 0.9}]
        samples.append({**s, "output": out})
    return {**record, "samples": samples}


def test_gemini_is_skipped_when_muse_samples_agree_above_the_gate(records):
    gate = config(k=3, gemini_below=0.8)
    assert evaluator.route(_agreeing(records[0], True), gate)["gemini"] is None
    assert evaluator.route(_agreeing(records[0], False), gate)["gemini"] is not None
    assert evaluator.route(_agreeing(records[0], True), config(k=3))["gemini"] is not None  # no gate: always


def test_bank_curation_adds_and_drops_phrases(records):
    record = records[0]
    routed = evaluator.route(record, config(bank={"add": [OFF_BANK[0]], "drop": ["call mom"]}))
    assert OFF_BANK[0] in routed["bank"]["phrases"]
    assert "call mom" not in routed["bank"]["phrases"] and "call mom" not in routed["bank"]["counts"]
    assert record["bank"]["phrases"][0] == "call mom"  # the stored record is never edited


def test_pool_cap_keeps_the_first_candidates_and_none(records):
    clips = evaluator.replay(records, config(pool_cap=3))
    assert all(len(c.pool) <= 3 and c.pool[-1] == "∅" for c in clips)


def test_dropped_feature_is_zero_and_prior_strength_scales_the_prior(records):
    plain = evaluator.replay(records, config())
    dropped = evaluator.replay(records, config(drop=["vote"]))
    strong = evaluator.replay(records, config(prior_strength=2.0))
    assert all(np.all(c.features[:, COL["vote"]] == 0) for c in dropped)
    for a, b in zip(plain, strong):
        np.testing.assert_allclose(b.features[:, COL["prior"]], 2.0 * a.features[:, COL["prior"]])


# ---------------------------------------------------------------- failures


def test_failures_are_the_given_clips_where_vtt_differs_from_stt(records):
    found = evaluator.failures(records[:20], config())
    ids = {r["id"] for r in records[:20]}
    assert found and all(f["id"] in ids for f in found)
    for f in found:
        assert f["vtt"] != f["stt"]
        assert 1 <= len(f["candidates"]) <= 5 and {"text", "p"} == set(f["candidates"][0])
        assert isinstance(f["stt_in_pool"], bool)


def test_a_clip_read_correctly_is_not_a_failure():
    clean = world(n=40, seed=5, muse=6.0, closed=6.0, gemini=6.0, miss_rate=0.0)
    found = evaluator.failures(clean, config(k=5))
    assert len(found) < 10  # strong signal: nearly every clip is read right after the first few


# ---------------------------------------------------------------- the production path: experiment.evaluate


def test_experiment_path_routes_records_and_passes_only_its_knobs(monkeypatch, records):
    seen = {}

    def fake_evaluate(recs, cfg, split):
        seen.update(records=recs, config=cfg, split=split)
        return {**GOOD, "gemini_acc": 0.2}

    monkeypatch.setitem(sys.modules, "experiment", types.SimpleNamespace(evaluate=fake_evaluate))
    c = config(k=2, sources=["open", "closed"], drop=["vote"], pool_cap=10)
    m = evaluator.ExperimentEvaluator().evaluate(records, c, "heldout")
    assert set(m) == set(evaluator.METRICS)
    assert set(seen["config"]) == {"costs", "drop", "stages", "pool_cap"}
    assert seen["config"]["drop"] == ["vote"] and seen["config"]["pool_cap"] == 10
    assert seen["split"] == "heldout"
    assert all(len(r["samples"]) == 2 and r["gemini"] is None for r in seen["records"])


def test_experiment_path_declares_knobs_it_cannot_evaluate():
    ev = evaluator.ExperimentEvaluator()
    assert ev.unsupported(config()) is None
    assert ev.unsupported(config(prior_strength=2.0))
    assert ev.unsupported(config(prompt_hint="watch the lips"))
