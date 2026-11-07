"""tools/bench.py: closed-set benchmark loaders (CREMA-D, GRID), label mapping and metrics. No network."""

import pytest

from tests.fakes import v1_output
from tools import bench


# ---------------------------------------------------------------- loaders and labels


def test_crema_filename_maps_to_actor_sentence_and_emotion():
    item = bench.crema_item("/x/1002_TIE_SAD_XX.flv")
    assert item == {"clip": "/x/1002_TIE_SAD_XX.flv", "text": "That is exactly what happened", "speaker": "1002",
                    "emotion": "SAD"}


def test_crema_has_the_twelve_sentences():
    assert len(bench.CREMA_SENTENCES) == 12
    assert bench.CREMA_SENTENCES["IEO"] == "It's eleven o'clock"
    assert bench.CREMA_SENTENCES["WSI"] == "We'll stop in a couple of minutes"


def test_crema_unknown_sentence_code_is_refused():
    with pytest.raises(ValueError):
        bench.crema_item("/x/1002_XYZ_SAD_XX.flv")


def test_grid_align_drops_silence_and_short_pauses():
    align = "0 23750 sil\n23750 29500 bin\n29500 34000 blue\n34000 35500 sp\n35500 41000 at\n41000 47250 f\n" \
            "47250 53000 two\n53000 61000 now\n61000 74500 sil\n"
    assert bench.grid_sentence(align) == "bin blue at f two now"


def test_grid_distractors_are_grammatical_new_and_change_exactly_two_slots_of_a_present_sentence():
    present = ["bin blue at f two now", "lay red by g nine soon", "set white with s one again"]
    distractors = bench.grid_distractors(present, 9, seed=3)
    assert len(distractors) == len(set(distractors)) == 9
    for d in distractors:
        words = d.split()
        assert d not in present and len(words) == 6
        assert all(w in options for w, options in zip(words, bench.GRID_GRAMMAR))
        assert min(sum(a != b for a, b in zip(words, p.split())) for p in present) == 2


def test_grid_distractors_are_seeded():
    present = ["bin blue at f two now", "lay red by g nine soon"]
    assert bench.grid_distractors(present, 5, seed=1) == bench.grid_distractors(present, 5, seed=1)


# ---------------------------------------------------------------- metrics


def test_topk_credit_counts_a_unique_winner():
    scores = {"a": 0.9, "b": 0.5, "c": 0.1}
    assert bench.topk_credit(scores, "a", 1) == 1.0
    assert bench.topk_credit(scores, "b", 1) == 0.0
    assert bench.topk_credit(scores, "b", 3) == 1.0


def test_topk_credit_splits_ties_as_a_random_tie_break_would():
    scores = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.0}
    assert bench.topk_credit(scores, "a", 1) == pytest.approx(0.25)
    assert bench.topk_credit(scores, "b", 3) == pytest.approx(0.75)
    assert bench.topk_credit(scores, "e", 3) == 0.0


def test_topk_credit_matches_text_normalized_and_a_failed_call_is_a_miss():
    assert bench.topk_credit({"It's eleven o'clock": 0.9, "x": 0.1}, "its eleven oclock", 1) == 1.0
    assert bench.topk_credit(None, "a", 3) == 0.0


def test_slot_hits_scores_each_grid_slot_by_alignment():
    assert bench.slot_hits("bin blue at f two now", "bin blue at f two now") == [1, 1, 1, 1, 1, 1]
    assert bench.slot_hits("bin red at f two now", "bin blue at f two now") == [1, 0, 1, 1, 1, 1]
    # a dropped word shifts nothing: the remaining words still align to their slots
    assert bench.slot_hits("bin blue f two now", "bin blue at f two now") == [1, 1, 0, 1, 1, 1]
    assert bench.slot_hits(None, "bin blue at f two now") == [0] * 6


def _record(user, truth, open_text, closed, gemini, phrases):
    out = v1_output((open_text, 0.8))
    samples = [{"call": "open", "sample_idx": i, "latency_ms": 1, "output": out, "error": None} for i in range(3)]
    return {"record": "v1", "id": f"{user}-{truth}", "user_id": user, "phrases": phrases,
            "label": {"stt_text": truth, "user_edit": None}, "samples": samples,
            "closed": {"call": "closed", "output": {"scores": closed, "none": 0.0}, "error": None},
            "gemini": {"call": "gemini", "output": gemini, "error": None}}


PHRASES = ["yes", "no", "maybe", "stop"]


def test_reads_report_each_reader_against_the_label():
    record = _record("u", "no", "yes", {"yes": 0.2, "no": 0.9, "maybe": 0.1, "stop": 0.0},
                     {"yes": 0.1, "no": 0.2, "maybe": 0.6, "stop": 0.1}, PHRASES)
    [row] = bench.reads([record])
    assert row["open_pick"] == "yes" and row["open_exact"] is False
    assert row["closed_top1"] == 1.0
    assert row["gemini_top1"] == 0.0 and row["gemini_top3"] == 1.0


def test_streams_train_only_on_earlier_clips_and_tag_the_stage():
    records = [_record("u", p, p, {q: float(q == p) for q in PHRASES}, {q: 0.25 for q in PHRASES}, PHRASES)
               for p in PHRASES * 9]
    rows = bench.stream(bench.reads(records))
    assert [r["n_before"] for r in rows] == list(range(len(records)))
    assert rows[5]["stage"] == "defaults" and rows[12]["stage"] == "temperature" and rows[31]["stage"] == "core"
    assert all(r["fused_defaults"] for r in rows)  # the closed call alone names every clip right


def test_summary_pairs_each_stage_with_defaults_on_the_same_clips():
    rows = [{"n_before": i, "fused_defaults": i % 2 == 0, "fused_temperature": i >= 10, "fused_core": i >= 30,
             "sent": i >= 20, "stage": "x", "open_exact": False, "open_words": (0, 6), "closed_top1": 0.0,
             "closed_top3": 0.0, "gemini_top1": 0.0, "gemini_top3": 0.0, "slots": None} for i in range(40)]
    summary = bench.summarize(rows)
    assert summary["n"] == 40
    assert summary["fused_defaults"] == pytest.approx(0.5)
    assert summary["fused_10"] == pytest.approx(1.0) and summary["fused_10_defaults"] == pytest.approx(0.5)
    assert summary["fused_30"] == pytest.approx(1.0) and summary["n_30"] == 10
    assert summary["autosend_coverage"] == pytest.approx(20 / 30) and summary["autosend_precision"] == 1.0
    assert summary["open_word_acc"] == 0.0
