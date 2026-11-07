"""tools/exp_crop.py: crop-variant boxes, per-turn read metrics, paired flips and the run record the Runs tab shows.

Box numbers are derived by hand from a tracker median box [100, 200, 90, 50]: centre (145, 225)."""

import importlib.util
import re
from pathlib import Path

import pytest

import visemes

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("exp_crop_tool", ROOT / "tools" / "exp_crop.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

BOX = [100, 200, 90, 50]


# ---------------------------------------------------------------- arms and boxes


def test_default_arm_box_is_the_tracker_box():
    assert tool.variant_box(BOX) == pytest.approx(BOX)


def test_scale_grows_both_sides_around_the_centre():
    # w 180, h 100 around (145, 225)
    assert tool.variant_box(BOX, scale=2) == pytest.approx([55, 175, 180, 100])


def test_sx_and_sy_scale_width_and_height_separately():
    # wide only: w 180, h 50
    assert tool.variant_box(BOX, sx=2, sy=1) == pytest.approx([55, 200, 180, 50])


def test_aspect_sets_the_height_from_the_scaled_width():
    # square at the same width: h 90 around y 225 -> y 180
    assert tool.variant_box(BOX, aspect=1) == pytest.approx([100, 180, 90, 90])
    # 1.5x wide at 4:3: w 135, h 101.25 -> x 145 - 67.5, y 225 - 50.625
    assert tool.variant_box(BOX, scale=1.5, aspect=4 / 3) == pytest.approx([77.5, 174.375, 135, 101.25])


def test_dx_moves_the_centre_by_a_fraction_of_the_tracker_width():
    # -0.1 * 90 = 9 px left
    assert tool.variant_box(BOX, dx=-0.1) == pytest.approx([91, 200, 90, 50])


def test_arm_text_is_an_optional_label_and_its_knobs():
    assert tool.parse_arm("x2:scale=2") == ("x2", {"scale": 2.0})
    assert tool.parse_arm("sx=2,sy=1") == ("sx=2,sy=1", {"sx": 2.0, "sy": 1.0})
    assert tool.parse_arm("base") == ("base", {})


def test_unknown_arm_knob_is_refused():
    with pytest.raises(ValueError):
        tool.parse_arm("x:zoom=2")


def test_coverage_counts_frames_whose_unpadded_mouth_stays_inside_the_crop():
    # pad 1.6: padded [10, 10, 32, 16] -> mouth 20 x 10 at centre (26, 18) -> [16, 13, 20, 10], inside [0, 0, 100, 100]
    # padded [90, 10, 32, 16] -> mouth [96, 13, 20, 10], right edge 116 > 100: outside
    boxes = [[0, 10, 10, 32, 16], [40, 90, 10, 32, 16]]
    assert tool.coverage(boxes, [0, 0, 100, 100], pad=1.6) == pytest.approx(0.5)


# ---------------------------------------------------------------- read metrics


def _sample(top):
    return {"output": {"hypotheses": [{"text": top, "s": 0.9}]}, "error": None}


FAILED = {"output": None, "error": "timeout"}


def test_turn_read_is_the_most_common_top_hypothesis():
    row = tool.turn_row([_sample("call mom"), _sample("we need more"), _sample("We need more."), FAILED],
                        "we need more", ["go"])
    assert row["pick"] == "we need more"
    assert tool.summarize([row])["viseme"] == pytest.approx(1.0)


def test_mean_of_k_averages_the_valid_samples_and_best_of_k_takes_the_closest():
    # one exact read, one empty read (every STT viseme deleted), one failed call (no read, not counted)
    row = tool.turn_row([_sample("we need more"), _sample(""), FAILED], "we need more", ["go"])
    summary = tool.summarize([row])
    assert summary["viseme_mean_k"] == pytest.approx(0.5)
    assert summary["viseme_best_k"] == pytest.approx(1.0)


def test_accuracies_pool_edits_over_turns_not_average_per_turn_scores():
    exact = tool.turn_row([_sample("we need more")], "we need more", ["go"])
    empty = tool.turn_row([FAILED], "go", ["we need more"])
    long, short = len(visemes.to_visemes("we need more")), len(visemes.to_visemes("go"))
    assert long != short
    assert tool.summarize([exact, empty])["viseme"] == pytest.approx(1 - short / (long + short))


def test_a_read_that_fits_every_turn_equally_has_no_specificity():
    same = tool.turn_row([_sample("we need more")], "we need more", ["we need more"])
    other = tool.turn_row([_sample("we need more")], "we need more", ["go fishing tomorrow afternoon"])
    assert tool.summarize([same])["specificity"] == pytest.approx(0.0)
    assert tool.summarize([other])["specificity"] > 0.3


def test_content_hits_count_shared_content_words_not_function_words():
    assert tool.content_hits("we will create jobs", "they create new jobs") == 2
    assert tool.content_hits("we will", "we will") == 0


def test_flips_count_per_turn_wins_losses_and_ties_with_a_sign_test():
    stts = ["we need more", "call mom now", "go fishing"]
    base = [tool.turn_row([_sample("")], s, ["x"]) for s in stts]
    arm = [tool.turn_row([_sample(stts[0])], stts[0], ["x"]), tool.turn_row([_sample(stts[1])], stts[1], ["x"]),
           tool.turn_row([_sample("")], stts[2], ["x"])]
    assert tool.flips(base, arm, "viseme") == {"up": 2, "down": 0, "ties": 1, "p": pytest.approx(0.5)}


# ---------------------------------------------------------------- run record


def _record(**overrides):
    stts = ["we need more", "call mom now"]
    rows = [tool.turn_row([_sample(s)], s, ["x"]) for s in stts]
    base = [tool.turn_row([_sample("")], s, ["x"]) for s in stts]
    options = dict(experiment="r1", arm="x2", spec={"scale": 2.0}, user="practice:obama-weekly:obama", rows=rows,
                   base_rows=base, turns=["c1", "c2"], reads=["n1", "n2"], hypothesis="h", verdict="inconclusive",
                   reopen="more turns", cost=0.5)
    return tool.run_record(**{**options, **overrides})


def test_run_record_has_what_the_runs_tab_reads():
    record = _record()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", record["id"])
    assert record["author"] == "agent:lab-crop" and record["loop"] == "framework"
    assert record["metrics"]["clips"] == 2
    assert [point["i"] for point in record["curve"]] == [0, 1]
    assert record["curve"][-1]["vtt"] == pytest.approx(record["metrics"]["vtt_muse_viseme_acc"])
    assert record["flips"]["viseme"]["up"] == 2


def test_run_record_refuses_an_unknown_verdict():
    with pytest.raises(ValueError):
        _record(verdict="win")
