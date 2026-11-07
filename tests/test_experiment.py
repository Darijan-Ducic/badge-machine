"""Offline experiment harness: baseline vs variant replayed on the same raw clips in the same order.

Simulated raw outputs go through the REAL featurizer with the REAL viseme functions (`visemes.to_visemes` for the
simulated observations too), so the viseme-cost knob acts on real codes.

Splits are chronological (docs/self-improvement.md): train = first 60% (predict-then-train), heldout = next 20%
(scored by one model trained on train: the gate), test = last 20% (scored by one model trained on train + heldout).

Reference points measured on these simulator worlds with candidates + scorer alone (no harness), which the
thresholds below lean on:
  "visemes carry the signal" (muse/closed/gemini 0.5, visemes 0.8), seeds 0-3:
    140 clips, model on the first 84 (full stage), heldout VTT 1.0; with `viseme_sim` zeroed 0.25-0.54.
    60 clips, model on the first 36: core stage (default plan) 0.17-0.5; full stage 1.0.
    70 clips: the hand-set default weights score 0.29-0.36 on clips 42-55 and 0.07-0.36 on clips 56-69.
  strong signals (muse/closed/gemini 3), 100 clips: a model on the first 80 auto-sends all of the last 20, all right.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import visemes
from candidates import featurize, match
from experiment import compare, config_for, evaluate, run
from readiness import readiness
from scorer import predict, prequential, train
from tests.simulator import simulate

VISEME_WORLD = dict(muse=0.5, closed=0.5, gemini=0.5, visemes=0.8, to_visemes=visemes.to_visemes)
CONTRACT = {"id", "t", "name", "dataset", "user_id", "config", "baseline", "metrics", "gate", "curve"}
NOTES = {"hypothesis", "author", "code_commit", "system_version", "candidate_version", "parent_version"}
METRICS = {"clips", "vtt_acc", "vtt_acc_heldout", "gemini_acc", "muse_acc", "pool_recall", "autosend_precision",
           "autosend_coverage"}
SPLIT_METRICS = {"clips", "word_acc", "sentence_acc", "viseme_acc", "top3_acc", "gemini_acc", "muse_acc",
                 "pool_recall", "autosend_precision", "autosend_coverage", "cost_per_clip"}
FLAT_COSTS = {"alphabet": "visemes-v1", "default_sub": 0, "ins": 0, "del": 0, "sub": {}}  # every distance 0


@pytest.fixture(scope="module")
def world140():
    return simulate(4, 140, **VISEME_WORLD)


@pytest.fixture(scope="module")
def world60():
    return simulate(0, 60, **VISEME_WORLD)


@pytest.fixture(scope="module")
def small():
    return simulate(21, 20, muse=1.5, closed=1.5, gemini=1.5, visemes=0.5, miss_rate=0.2,
                    to_visemes=visemes.to_visemes)


def clips_of(records):
    """Independent featurization with the default knobs (real visemes, data/viseme_costs.json, cap 24)."""
    return [featurize(**r, to_visemes=visemes.to_visemes, viseme_distance=visemes.viseme_distance) for r in records]


def vtt_correct(clip, probs):
    top = int(probs.argmax())
    return top != len(clip.pool) - 1 and match(clip.stt, [clip.pool[top]]) is not None


def defaults_accuracy(clips):
    return np.mean([vtt_correct(c, p) for c, p in zip(clips, predict(train([]), clips))])


# ---- the gate: same clips, same order, strictly better heldout VTT ----

def test_identical_configs_give_identical_metrics_and_fail_the_gate(world140):
    base, same = compare(world140, {"same": {}}, dataset="sim:viseme-world:4")

    assert same["metrics"] == base["metrics"]
    assert same["splits"] == base["splits"]
    assert same["curve"] == base["curve"]
    assert same["config"]["hash"] == base["config"]["hash"]
    assert same["gate"] == "fail"  # no improvement is not a pass
    assert (same["baseline"], base["baseline"], base["gate"]) == (base["id"], None, None)


def test_dropping_the_feature_that_carries_the_signal_lowers_heldout_and_fails(world140):
    base, dropped = compare(world140, {"no-visemes": {"drop": ["viseme_sim"]}}, dataset="sim:viseme-world:4")

    assert base["metrics"]["vtt_acc_heldout"] >= 0.9
    assert dropped["metrics"]["vtt_acc_heldout"] < base["metrics"]["vtt_acc_heldout"] - 0.3
    assert dropped["gate"] == "fail"


def test_fitting_all_features_earlier_raises_heldout_and_passes(world60):
    early_full = {"stages": {"defaults": 0, "temperature": 10, "full": 30}}  # viseme_sim gets weight from 30 labels

    base, variant = compare(world60, {"early-full": early_full}, dataset="sim:viseme-world:0")

    assert variant["metrics"]["vtt_acc_heldout"] > base["metrics"]["vtt_acc_heldout"] + 0.3
    assert variant["gate"] == "pass"


def test_flat_viseme_costs_erase_the_viseme_signal(world140):
    base, flat = compare(world140, {"flat-costs": {"costs": FLAT_COSTS}}, dataset="sim:viseme-world:4")

    assert flat["metrics"]["vtt_acc_heldout"] < base["metrics"]["vtt_acc_heldout"] - 0.3
    assert flat["gate"] == "fail"


def test_a_baseline_run_on_other_clips_is_refused(small):
    base = run(small, dataset="sim:small", name="baseline")

    with pytest.raises(ValueError):
        run(small[::-1], dataset="sim:small", name="reordered", baseline=base)
    with pytest.raises(ValueError):
        run(small[:-1], dataset="sim:small", name="fewer clips", baseline=base)
    with pytest.raises(ValueError):
        run(small, dataset="sim:other", name="other dataset", baseline=base)


# ---- splits: each one scored only by what came before it ----

def test_heldout_is_scored_by_a_model_trained_only_on_the_train_split():
    # 70 clips -> train 0-41, heldout 42-55. With `full` from 43 labels on, a model trained on the 42 train clips is
    # still at the hand-set defaults; any leaked heldout clip would switch it to the full fit (which scores 1.0 here).
    records = simulate(1, 70, **VISEME_WORLD)
    expected = defaults_accuracy(clips_of(records)[42:56])
    assert expected < 0.6  # so a leak would show

    metrics = evaluate(records, {"stages": {"defaults": 0, "full": 43}}, "heldout")

    assert metrics["clips"] == 14
    assert metrics["sentence_acc"] == pytest.approx(expected)
    assert (metrics["autosend_coverage"], metrics["autosend_precision"]) == (0.0, None)  # defaults never send


def test_test_split_is_scored_by_a_model_trained_only_on_train_and_heldout():
    records = simulate(1, 70, **VISEME_WORLD)
    expected = defaults_accuracy(clips_of(records)[56:70])
    assert expected < 0.6

    metrics = evaluate(records, {"stages": {"defaults": 0, "full": 57}}, "test")

    assert metrics["clips"] == 14
    assert metrics["sentence_acc"] == pytest.approx(expected)


def test_train_split_is_predict_then_train_over_the_first_60_percent(small):
    history = prequential(clips_of(small)[:12])

    metrics = evaluate(small, {}, "train")

    assert metrics["clips"] == 12
    assert metrics["sentence_acc"] == pytest.approx(np.mean([h["fused"] for h in history]))
    assert metrics["gemini_acc"] == pytest.approx(np.mean([h["gemini"] for h in history]))
    assert metrics["muse_acc"] == pytest.approx(np.mean([h["muse"] for h in history]))
    assert metrics["pool_recall"] == pytest.approx(np.mean([h["pool_hit"] for h in history]))
    assert (metrics["autosend_precision"], metrics["autosend_coverage"]) == (None, None)  # trained without a gate


def test_train_stream_scores_each_clip_before_learning_its_label():
    # Weak signals, and every feature fit from 1 label on: a model that already saw a clip memorizes it (measured on
    # seeds 0-2: 4-6 of 24 clips flip to right), so scoring with it would show up in the per-clip curve.
    records = simulate(1, 40, muse=0.3, closed=0.3, gemini=0.3, to_visemes=visemes.to_visemes)
    plan = {"defaults": 0, "full": 1}
    history = prequential(clips_of(records)[:24], stages=[(1, "full"), (0, "defaults")])

    curve = run(records, {"stages": plan}, dataset="sim:viseme-world:1", name="eager")["curve"]

    assert [point["vtt"] for point in curve] == pytest.approx(
        [np.mean([h["fused"] for h in history[max(0, i - 19):i + 1]]) for i in range(24)])


def test_every_split_reports_every_metric(small):
    for split in ("train", "heldout", "test"):
        assert set(evaluate(small, {}, split)) == SPLIT_METRICS


def test_top3_holds_every_candidate_when_the_pool_has_three():
    # Pool misses with bad-quality reports teach the model to rank ∅ high; ∅ must not take a top-3 place.
    records = simulate(7, 60, muse=1, closed=1, gemini=1, visemes=0.3, miss_rate=0.3, to_visemes=visemes.to_visemes)

    capped = evaluate(records, {"pool_cap": 4}, "heldout")  # 3 real candidates + ∅
    full = evaluate(records, {}, "heldout")

    assert capped["top3_acc"] == pytest.approx(capped["pool_recall"])
    assert full["sentence_acc"] <= full["top3_acc"] <= full["pool_recall"]


def test_a_runs_stored_config_reevaluates_to_its_split_metrics(small):
    record = run(small, {"drop": ["prior"], "pool_cap": 10}, dataset="sim:small", name="v")

    assert evaluate(small, record["config"], "heldout") == record["splits"]["heldout"]


def test_unknown_split_is_refused(small):
    with pytest.raises(ValueError):
        evaluate(small, {}, "validation")


# ---- the run record's headline metrics ----

def test_headline_metrics_come_from_the_train_stream_and_the_heldout_split(small):
    record = run(small, dataset="sim:small", name="baseline")
    train_split, heldout = record["splits"]["train"], record["splits"]["heldout"]

    assert set(record["splits"]) == {"train", "heldout"}  # the final test split is only evaluated on request
    assert record["metrics"] == {
        "clips": 20,
        "vtt_acc": train_split["sentence_acc"],
        "vtt_acc_heldout": heldout["sentence_acc"],
        "gemini_acc": train_split["gemini_acc"],
        "muse_acc": train_split["muse_acc"],
        "pool_recall": train_split["pool_recall"],
        "autosend_precision": heldout["autosend_precision"],
        "autosend_coverage": heldout["autosend_coverage"],
    }


def test_curve_is_rolling_accuracy_over_the_last_20_train_clips(world60):
    history = prequential(clips_of(world60)[:36])

    curve = run(world60, dataset="sim:viseme-world:0", name="baseline")["curve"]

    assert [point["i"] for point in curve] == list(range(36))
    for i in range(36):
        window = history[max(0, i - 19):i + 1]
        for key, name in (("vtt", "fused"), ("gemini", "gemini"), ("muse", "muse")):
            assert curve[i][key] == pytest.approx(np.mean([h[name] for h in window]))


def test_clips_without_stt_are_not_scored(small):
    unlabeled = [{**r, "stt": None} for r in simulate(22, 3, to_visemes=visemes.to_visemes)]

    with_unlabeled = run(small + unlabeled, dataset="sim:small+", name="baseline")
    labeled_only = run(small, dataset="sim:small", name="baseline")

    assert with_unlabeled["metrics"] == labeled_only["metrics"]
    assert with_unlabeled["splits"] == labeled_only["splits"]


def test_strong_signals_auto_send_precisely():
    metrics = run(simulate(2, 100, muse=3, closed=3, gemini=3, to_visemes=visemes.to_visemes),
                  dataset="sim:strong", name="baseline")["metrics"]

    assert metrics["autosend_coverage"] > 0.5
    assert metrics["autosend_precision"] >= 0.9


def test_pool_cap_keeps_only_the_first_candidates(small):
    first_is_right = np.mean([c.label == 0 for c in clips_of(small)[:12]])

    capped = run(small, {"pool_cap": 2}, dataset="sim:small", name="one candidate")  # one real candidate + ∅

    assert capped["metrics"]["pool_recall"] == pytest.approx(first_is_right)


def test_readiness_uses_the_runs_knobs_on_the_train_stream(small):
    history = prequential(clips_of(small)[:12])
    stts = [r["stt"] for r in small[:12]]
    knobs = {"n_min": 1, "window": 3, "acc_min": 0.5}

    default = run(small, dataset="sim:small", name="baseline")
    relaxed = run(small, {"readiness": knobs}, dataset="sim:small", name="relaxed")

    assert default["readiness"] == readiness(history, stts)
    assert relaxed["readiness"] == readiness(history, stts, **knobs)
    assert relaxed["readiness"] != default["readiness"]


def test_too_few_clips_for_three_splits_is_refused(small):
    with pytest.raises(ValueError):
        run(small[:2], dataset="sim:tiny", name="baseline")


# ---- the record ----

def test_record_matches_the_runs_contract(small):
    base, variant = compare(
        small, {"no-prior": {"drop": ["prior"]}}, dataset="sim:small", user_id="practice:sim", author="agent",
        system_version="sys-v1",
        variant_notes={"no-prior": {"hypothesis": "the prior is noise on simulated data", "candidate_version": "sys-v2",
                                    "parent_version": "sys-v1"}})

    for record in (base, variant):
        assert CONTRACT | NOTES | {"splits", "readiness"} == set(record)
        assert set(record["metrics"]) == METRICS
        assert isinstance(record["metrics"]["clips"], int)
        assert all(isinstance(record["metrics"][k], float) for k in METRICS - {"clips", "autosend_precision"})
        assert all(set(point) == {"i", "vtt", "gemini", "muse"} for point in record["curve"])
        assert (record["dataset"], record["user_id"], record["system_version"], record["author"]) == (
            "sim:small", "practice:sim", "sys-v1", "agent")
        json.dumps(record)  # stores as plain JSON (no numpy scalars)
    assert (base["name"], variant["name"]) == ("baseline", "no-prior")
    assert (base["hypothesis"], base["candidate_version"], base["parent_version"]) == ("", None, None)
    assert (variant["hypothesis"], variant["candidate_version"], variant["parent_version"]) == (
        "the prior is noise on simulated data", "sys-v2", "sys-v1")
    assert variant["gate"] in ("pass", "fail")
    assert variant["id"] != base["id"]


def test_record_stores_the_code_commit(small):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                          cwd=Path(__file__).parent).stdout.strip()

    assert run(small, dataset="sim:small", name="baseline")["code_commit"] == head


def test_author_is_agent_or_human(small):
    with pytest.raises(ValueError):
        run(small, dataset="sim:small", name="baseline", author="robot")


# ---- the config hash: every knob, the dataset id and the clip order; nothing else ----

def hash_of(records, spec=None, dataset="sim:small"):
    return config_for(records, spec or {}, dataset)["hash"]


@pytest.mark.parametrize("spec", [
    {"costs": {**FLAT_COSTS, "default_sub": 1, "ins": 1, "del": 1, "sub": {"P|F": 0.5}}},
    {"drop": ["prior"]},
    {"stages": {"defaults": 0, "temperature": 10, "full": 30}},
    {"pool_cap": 12},
    {"readiness": {"n_min": 3}},
    {"readiness": {"window": 4}},
    {"readiness": {"acc_min": 0.9}},
], ids=["costs", "drop", "stages", "pool_cap", "n_min", "window", "acc_min"])
def test_hash_changes_with_each_knob(small, spec):
    assert hash_of(small, spec) != hash_of(small)


def test_hash_changes_with_the_dataset_id_and_clip_order(small):
    swapped = [small[1], small[0], *small[2:]]

    assert hash_of(small, dataset="sim:other") != hash_of(small)
    assert hash_of(swapped) != hash_of(small)
    assert hash_of(small[:-1]) != hash_of(small)


def test_hash_ignores_how_the_same_config_is_written(small, tmp_path):
    costs_file = tmp_path / "costs.json"
    costs_file.write_text(json.dumps({"_doc": "a note", **FLAT_COSTS}))
    defaults_spelled_out = {"costs": visemes.load_costs(), "drop": [], "pool_cap": 24,
                            "stages": {"full": 60, "core": 30, "temperature": 10, "defaults": 0},
                            "readiness": {"n_min": 5, "window": 5, "acc_min": 0.8}}

    assert hash_of(small, defaults_spelled_out) == hash_of(small)
    assert hash_of(small, {"drop": ["prior", "gemini"]}) == hash_of(small, {"drop": ["gemini", "prior"]})
    assert hash_of(small, {"costs": str(costs_file)}) == hash_of(small, {"costs": FLAT_COSTS})


def test_record_hash_ignores_the_run_notes(small):
    first = run(small, dataset="sim:small", name="a", hypothesis="one idea", author="human")
    second = run(small, dataset="sim:small", name="b", hypothesis="another idea", author="agent",
                 system_version="sys-v2", code_commit="deadbeef", user_id="someone", candidate_version="sys-v3",
                 parent_version="sys-v2")

    assert first["config"]["hash"] == second["config"]["hash"] == hash_of(small)


def test_unknown_knobs_are_refused(small):
    with pytest.raises(ValueError):
        config_for(small, {"pool_capp": 12}, "sim:small")
    with pytest.raises(ValueError):
        config_for(small, {"drop": ["not_a_feature"]}, "sim:small")
    with pytest.raises(ValueError):
        config_for(small, {"stages": {"full": 30}}, "sim:small")  # no stage from 0 labels
    with pytest.raises(ValueError):
        config_for(small, {"pool_cap": 30}, "sim:small")  # above candidates.POOL_CAP
