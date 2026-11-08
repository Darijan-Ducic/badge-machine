"""Candidate pool, features and label on one small hand-written clip; every expected value is derived in comments.

Viseme functions are the FAKES from tests/simulator.py: each letter is a code (repeats collapsed), distance is
Levenshtein / longer length. Hand-derived viseme strings used below:
  "go on" -> g o n      "go up" -> g o u p     "no" -> n o       "stop" -> s t o p
  "help me" -> h e l p m e                      "wait" -> w a i t "go" -> g o
"""

import math

import pytest

from candidates import FEATURES, NONE, featurize
from tests.simulator import fake_to_visemes, fake_viseme_distance


def record(idx, output, error=None):
    return {"call": "open", "sample_idx": idx, "latency_ms": 2000, "output": output, "error": error}


def visemes(codes):
    return [{"v": v, "f": [i, i]} for i, v in enumerate(codes)]


def slot(*alts):
    return {"f": [0, 0], "alts": [{"w": w, "s": s} for w, s in alts]}


def hyps(*pairs):
    return [{"text": t, "s": s} for t, s in pairs]


# S0: clean view. Observed visemes: REST dropped, "o o" collapsed -> g o n.
S0 = record(0, {
    "schema_version": "v1",
    "speech": {"present": True, "frames": [0, 20]},
    "quality": {"mouth_visible": "full", "blur": "low", "occluded_frames": []},
    "visemes": visemes(["REST", "g", "o", "o", "n", "REST"]),
    "slots": [slot(("go", 0.8), ("no", 0.2)), slot(("on", 0.5), ("up", 0.4), ("in", 0.1))],
    "hypotheses": hyps(("go on", 0.6), ("no", 0.3)),
})
# S1: mouth partly hidden (bad quality); lists "go on" twice (as "go on" and "Go on.").
S1 = record(1, {
    "schema_version": "v1",
    "speech": {"present": True, "frames": [0, 20]},
    "quality": {"mouth_visible": "partial", "blur": "low", "occluded_frames": [[5, 9]]},
    "visemes": visemes(["g", "o", "u", "p"]),
    "slots": [slot(("go", 0.9)), slot(("up", 0.6), ("on", 0.4), ("in", 0.2))],
    "hypotheses": hyps(("go up", 0.5), ("go on", 0.4), ("Go on.", 0.1)),
})
FAILED = record(2, None, error="timeout")
# S2: high blur (bad quality) and says no speech, yet still guesses.
S2 = record(3, {
    "schema_version": "v1",
    "speech": {"present": False, "frames": []},
    "quality": {"mouth_visible": "full", "blur": "high", "occluded_frames": []},
    "visemes": visemes(["h", "e", "l", "p"]),
    "slots": [slot(("no", 0.7), ("go", 0.2))],
    "hypotheses": hyps(("no", 0.7), ("go on", 0.2)),
})
SAMPLES = [S0, S1, FAILED, S2]
CLOSED = {"scores": {"go on": 0.5, "go up": 0.2, "no": 0.1, "stop": 0.15, "wait": 0.12, "help me": 0.08}, "none": 0.3}
GEMINI = {"go on": 0.5, "go up": 0.1, "no": 0.15, "stop": 0.25}
# Bank entries: 4 phrases + past STT texts "help me" and "wait" -> bank size 6; total count 3+1+2+1 = 7.
# prior denominator = total + bank size + 1 = 7 + 6 + 1 = 14.
BANK = {"phrases": ["go on", "go up", "no", "stop"], "counts": {"go on": 3, "no": 1, "help me": 2, "wait": 1}}

# Pool, in insertion order:
#   hypotheses:      go on, no (S0); go up (S1; "go on"/"Go on." are duplicates); S2 adds nothing
#   recombinations:  S0 top-2 alts per slot: go/no x on/up -> go on .40, go up .32, no on .10, (no up .08 dropped)
#                    -> adds "no on"; S1: go x up/on (its 3rd alt "in" is outside the top 2) -> go up .54, go on .36
#                    -> adds nothing; S2: no .7, go .2 -> adds "go"
#   closed top 3:    go on .5, go up .2, stop .15 -> adds "stop" (4th, "wait" .12, stays out)
#   Gemini top 3:    go on .5, stop .25, no .15 -> adds nothing
#   bank nearest 3:  min distance over samples (observed g o n / g o u p / h e l p):
#                    go on 0, go up 0, help me 2/6 (vs h e l p), no 2/3, stop 3/4, wait 1 -> adds "help me"
POOL = ["go on", "no", "go up", "no on", "go", "stop", "help me"]


def clip(samples=SAMPLES, stt=None, closed=CLOSED, gemini=GEMINI):
    return featurize(samples, closed, gemini, BANK, stt, to_visemes=fake_to_visemes,
                     viseme_distance=fake_viseme_distance)


def row(result, text):
    return dict(zip(FEATURES, result.features[result.pool.index(text)]))


def test_features_are_named_in_spec_order():
    assert FEATURES == ["vote", "top_vote", "mean_score", "slot_support", "viseme_sim", "closed", "gemini", "prior",
                        "in_bank", "pick_history", "is_none", "none_closed", "bad_quality", "no_speech"]


def test_pool_holds_every_source_deduped_with_none_last():
    result = clip()

    assert result.pool[-1] == NONE
    assert sorted(result.pool[:-1]) == sorted(POOL)
    assert result.features.shape == (len(POOL) + 1, len(FEATURES))


def test_features_of_a_candidate_every_source_backs():
    assert row(clip(), "go on") == pytest.approx({
        "vote": 1.0,                       # S0, S1 (once, despite "Go on."), S2: 3/3
        "top_vote": 1 / 3,                 # only S0's highest-s hypothesis
        "mean_score": 0.4,                 # (.6 + .4 + .2) / 3; S1's first exact match is .4
        "slot_support": 5 / 6,             # go+on in S0, go+on in S1, only go in S2: (1 + 1 + 1/2) / 3
        "viseme_sim": 0.5,                 # 1 - (0 [gon,gon] + 2/4 [gon,goup] + 4/4 [gon,help]) / 3
        "closed": 0.5,
        "gemini": 0.5,
        "prior": math.log(4 / 14),         # (3 + 1) / 14
        "in_bank": 1.0,
        "pick_history": 0.0,               # BANK has no "picks"
        "is_none": 0.0, "none_closed": 0.0, "bad_quality": 0.0, "no_speech": 0.0,
    })


def test_features_of_a_candidate_only_the_bank_and_visemes_back():
    assert row(clip(), "help me") == pytest.approx({
        "vote": 0.0, "top_vote": 0.0, "mean_score": 0.0,
        "slot_support": 0.0,               # neither word appears in any slot alt
        "viseme_sim": 5 / 18,              # 1 - (6/6 [helpme,gon] + 5/6 [helpme,goup] + 2/6 [helpme,help]) / 3
        "closed": 0.08,
        "gemini": 0.0,                     # not in the phrase list Gemini scored
        "prior": math.log(3 / 14),         # (2 + 1) / 14
        "in_bank": 0.0,                    # a past STT text, not a phrase-list entry
        "pick_history": 0.0,
        "is_none": 0.0, "none_closed": 0.0, "bad_quality": 0.0, "no_speech": 0.0,
    })


def test_features_of_a_slot_recombination_nobody_hypothesized():
    assert row(clip(), "go") == pytest.approx({
        "vote": 0.0, "top_vote": 0.0, "mean_score": 0.0,
        "slot_support": 1.0,               # "go" is an alt in every sample
        "viseme_sim": 7 / 18,              # 1 - (1/3 [go,gon] + 2/4 [go,goup] + 4/4 [go,help]) / 3
        "closed": 0.0, "gemini": 0.0,
        "prior": math.log(1 / 14),         # never said before
        "in_bank": 0.0, "pick_history": 0.0,
        "is_none": 0.0, "none_closed": 0.0, "bad_quality": 0.0, "no_speech": 0.0,
    })


def test_none_carries_the_clip_level_signals():
    assert row(clip(), NONE) == pytest.approx({
        "vote": 0.0, "top_vote": 0.0, "mean_score": 0.0, "slot_support": 0.0, "viseme_sim": 0.0,
        "closed": 0.0, "gemini": 0.0, "prior": 0.0, "in_bank": 0.0, "pick_history": 0.0,
        "is_none": 1.0,
        "none_closed": 0.3,
        "bad_quality": 2 / 3,              # S1 mouth partial, S2 blur high; S0 clean
        "no_speech": 1 / 3,                # S2
    })


def test_label_matches_stt_after_normalizing():
    assert clip(stt="Go on.").label == clip().pool.index("go on")
    assert clip(stt="go go on").label == clip().pool.index("go on")  # STT word doubling collapsed


def test_label_needs_an_exact_match_so_a_near_miss_is_a_pool_miss():
    # Spec change: fuzzy matching (ratio >= .85) merged opposite commands ("turn on/off the lights" = .92), so a
    # near miss is now a pool miss. "go onn" vs "go on" (ratio .91) no longer counts as "go on".
    assert clip(stt="go onn").label == clip().pool.index(NONE)


def test_near_identical_commands_stay_separate_candidates_with_their_own_votes():
    on = lambda i: record(i, {**S0["output"], "hypotheses": hyps(("turn on the lights", 0.6))})
    off = record(2, {**S0["output"], "hypotheses": hyps(("turn off the lights", 0.6))})
    c = clip(samples=[on(0), on(1), off], stt="turn off the lights")
    vote = FEATURES.index("vote")
    # 2 of 3 samples say "on", 1 of 3 says "off"; a fuzzy merge would give both 3/3
    assert c.features[c.pool.index("turn on the lights")][vote] == pytest.approx(2 / 3)
    assert c.features[c.pool.index("turn off the lights")][vote] == pytest.approx(1 / 3)
    assert c.label == c.pool.index("turn off the lights")


def test_label_is_none_when_stt_is_not_in_the_pool():
    pool = clip().pool
    assert clip(stt="turn it off").label == pool.index(NONE)
    # "go onto" vs "go on": ratio 2*5/12 = .83 -- below .85 (preheat.match_phrase's default .8 would accept it)
    assert clip(stt="go onto").label == pool.index(NONE)


def test_unlabeled_clip_has_no_label():
    assert clip().label is None


def test_failed_samples_are_skipped():
    with_failure, without = clip(stt="go on"), clip([S0, S1, S2], stt="go on")

    assert with_failure.pool == without.pool
    assert (with_failure.features == without.features).all()
    assert with_failure.label == without.label


def test_clip_with_no_valid_samples_still_gets_a_pool():
    result = clip([FAILED, FAILED])

    # closed top 3: go on, go up, stop; Gemini top 3: go on, stop, no; bank nearest is undefined without
    # observed visemes, so every entry ties and the first three in bank order are taken: go on, go up, no.
    assert result.pool[-1] == NONE
    assert sorted(result.pool[:-1]) == sorted(["go on", "go up", "stop", "no"])
    sample_features = ["vote", "top_vote", "mean_score", "slot_support", "viseme_sim", "bad_quality", "no_speech"]
    columns = [FEATURES.index(name) for name in sample_features]
    assert (result.features[:, columns] == 0).all()
    assert row(result, "stop")["closed"] == pytest.approx(0.15)
    assert row(result, NONE)["none_closed"] == pytest.approx(0.3)


def test_reports_each_models_own_pick():
    result = clip([S0, S2, S2])  # top hypotheses: go on, no, no

    assert result.gemini_pick == "go on"
    assert result.muse_pick == "no"
    assert clip([FAILED], gemini={}).muse_pick is None
    assert clip([FAILED], gemini={}).gemini_pick is None


def test_pool_is_capped_at_24_with_none_kept_last():
    many = record(0, {**S0["output"], "hypotheses": hyps(*[(f"phrase number {i}", 0.01) for i in range(40)])})

    result = clip([many])

    assert len(result.pool) == 24
    assert result.pool[-1] == NONE
    assert result.pool[:3] == ["phrase number 0", "phrase number 1", "phrase number 2"]


def test_phrase_list_candidates_keep_their_pool_slots_when_open_hypotheses_overflow_the_cap():
    # 40 distinct open hypotheses alone would take all 23 real slots. Closed top 3: go on, go up, stop; Gemini top 3:
    # go on, stop, no. Those 4 (plus the 3 bank neighbours) are reserved; open hypotheses fill the other slots.
    many = record(0, {**S0["output"], "hypotheses": hyps(*[(f"phrase number {i}", 0.01) for i in range(40)])})

    result = clip([many], stt="stop")

    assert len(result.pool) == 24 and result.pool[-1] == NONE
    assert {"go on", "go up", "stop", "no"} <= set(result.pool)
    assert result.pool[result.label] == "stop"
    assert result.pool[0] == "phrase number 0"


def test_an_open_hypothesis_that_is_also_reserved_does_not_use_up_an_open_slot():
    many = record(0, {**S0["output"], "hypotheses": hyps(("go on", 0.5),
                                                          *[(f"phrase number {i}", 0.01) for i in range(40)])})

    result = clip([many])

    assert len(result.pool) == 24
    assert result.pool[0] == "go on"


# Earlier takes, by Gemini's top pick then label: Gemini picked "go on" 4 times, and the user's label was "go on"
# once and "wait" 3 times; Gemini picked "stop" twice, both labeled "no". GEMINI's top pick here is "go on".
PICKS = {"go on": {"go on": 1, "wait": 2, "Wait.": 1}, "stop": {"no": 2}}


def picked(gemini=GEMINI):
    return featurize(SAMPLES, CLOSED, gemini, {**BANK, "picks": PICKS}, None, to_visemes=fake_to_visemes,
                     viseme_distance=fake_viseme_distance)


def test_pick_history_is_the_share_of_earlier_takes_with_the_same_gemini_pick_labeled_as_the_candidate():
    result = picked()

    assert row(result, "wait")["pick_history"] == pytest.approx(3 / 4)  # "Wait." counts as "wait"
    assert row(result, "go on")["pick_history"] == pytest.approx(1 / 4)
    assert row(result, "no")["pick_history"] == 0.0  # its history is under Gemini's "stop" pick, not this read's
    assert row(result, NONE)["pick_history"] == 0.0
    # "wait" is in no other source's top 3 (closed 4th at .12, see POOL); its history reserves it a slot
    assert "wait" not in POOL and "wait" in result.pool


def test_pick_history_is_zero_without_a_gemini_read():
    result = picked(gemini=None)

    assert (result.features[:, FEATURES.index("pick_history")] == 0).all()
    assert sorted(result.pool[:-1]) == sorted(POOL)
