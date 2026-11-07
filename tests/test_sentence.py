"""sentence.py: combine K open samples' word readings into one sentence (alignment, MBR, ROVER, slot rebuild).

Expected values are derived by hand in each test; the random checks compare against experiment._edits, the unit-cost
Levenshtein the word-accuracy metric itself uses, so an alignment that is not minimal cannot pass.
"""

import random

from experiment import _edits
from sentence import align, mbr, rover, slot_words, words

W = str.split


def test_align_pairs_equal_words_and_marks_insertions_and_deletions_as_gaps():
    assert align(W("we need to do more"), W("we really need to do")) == [
        ("we", "we"), (None, "really"), ("need", "need"), ("to", "to"), ("do", "do"), ("more", None)]


def test_align_substitutes_rather_than_paying_a_gap_on_each_side():
    assert align(W("a b c"), W("a x c")) == [("a", "a"), ("b", "x"), ("c", "c")]


def test_align_is_a_minimum_edit_alignment_that_keeps_both_sequences():
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 7))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 7))]
        pairs = align(a, b)
        assert [x for x, _ in pairs if x is not None] == a
        assert [y for _, y in pairs if y is not None] == b
        assert all(x is not None or y is not None for x, y in pairs)
        assert sum(x != y for x, y in pairs) == _edits(a, b)


def test_mbr_picks_the_hypothesis_closest_to_all_the_others_not_the_first():
    # word edits: h1-h2 1, h1-h3 1, h2-h3 2, h0-h1 3, h0-h2 3, h0-h3 4 -> sums h0 10, h1 5, h2 6, h3 7
    hyps = [W("x y"), W("a b c"), W("a b d"), W("a b c e")]
    assert mbr(hyps) == 1


def test_mbr_weights_scale_each_hypothesis_as_evidence():
    # weights 1,1,1,10 -> h0 3+3+40=46, h1 3+1+10=14, h2 3+1+20=24, h3 4+1+2=7
    hyps = [W("x y"), W("a b c"), W("a b d"), W("a b c e")]
    assert mbr(hyps, [1, 1, 1, 10]) == 3


def test_mbr_uses_the_given_distance():
    by_length = lambda a, b: abs(len(a) - len(b))  # noqa: E731
    # lengths 1, 3, 4, 4 -> sums h0 2+3+3=8, h1 2+1+1=4, h2 3+1+0=4, h3 4 (tie: first, h1); word edits pick h0
    assert mbr([W("a"), W("x y z"), W("p q r s"), W("t u v w")], distance=by_length) == 1


def test_mbr_counts_substituted_words_not_only_the_length_difference():
    # word edits: h0-h1 2, h0-h2 2, h1-h2 0 -> sums h0 4, h1 2, h2 2; a length-only distance would tie all at 0
    assert mbr([W("a b"), W("c d"), W("c d")]) == 1


def test_mbr_ties_go_to_the_first_hypothesis():
    assert mbr([W("a b"), W("c d")]) == 0


def test_rover_takes_the_majority_word_in_each_column():
    assert rover([W("a x c"), W("a b c"), W("a b c")]) == W("a b c")


def test_rover_drops_a_word_most_hypotheses_lack_unless_gaps_are_off():
    hyps = [W("a b"), W("a"), W("a")]
    assert rover(hyps) == W("a")
    assert rover(hyps, gaps=False) == W("a b")


def test_rover_one_inserted_word_does_not_shift_the_other_hypotheses():
    hyps = [W("we really need to do more"), W("we need to do more"), W("we need to see more")]
    assert rover(hyps) == W("we need to do more")


def test_rover_a_word_only_a_later_hypothesis_inserts_is_outvoted_by_the_earlier_ones():
    assert rover([W("we need"), W("we need"), W("we really need")]) == W("we need")
    assert rover([W("we need"), W("we really need"), W("we really need")]) == W("we really need")


def test_rover_weights_decide_a_column():
    assert rover([W("a b"), W("a c"), W("a c")], [3, 1, 1]) == W("a b")
    assert rover([W("a b"), W("a c"), W("a c")]) == W("a c")


def test_rover_ties_go_to_the_word_seen_first():
    assert rover([W("a b"), W("a c")]) == W("a b")


def test_rover_a_tie_between_a_word_and_a_gap_goes_to_whichever_came_first():
    assert rover([W("a b"), W("a")]) == W("a b")
    assert rover([W("a"), W("a b")]) == W("a")


def test_rover_a_word_one_of_three_hypotheses_says_does_not_survive():
    # "z" aligns with "a" for free (the "c" column already holds a gap), so "c" keeps 1 of 3 votes and loses
    assert rover([W("a c"), W("a"), W("z")]) == W("a")


def test_rover_of_one_hypothesis_is_that_hypothesis():
    assert rover([W("we need to do more")]) == W("we need to do more")
    assert rover([]) == []


def test_rover_only_outputs_words_some_hypothesis_has_in_that_order_of_some_column():
    rng = random.Random(1)
    for _ in range(200):
        hyps = [[rng.choice("abcdef") for _ in range(rng.randint(1, 6))] for _ in range(rng.randint(1, 5))]
        out = rover(hyps, [rng.random() + 0.1 for _ in hyps], gaps=rng.random() < 0.5)
        seen = {w for h in hyps for w in h}
        assert set(out) <= seen
        assert len(out) <= sum(len(h) for h in hyps)


def test_slot_words_takes_the_best_scored_alternative_of_each_slot_in_order():
    output = {"slots": [
        {"f": [0, 9], "alts": [{"w": "hi", "s": 0.25}, {"w": "I", "s": 0.3}]},
        {"f": [10, 12], "alts": []},
        {"f": [13, 21], "alts": [{"w": "want", "s": 0.2}, {"w": "Love you,", "s": 0.28}]},
    ]}
    assert slot_words(output) == W("i love you")


def test_slot_words_of_an_output_without_slots_is_empty():
    assert slot_words({"hypotheses": [{"text": "hi", "s": 1}]}) == []


def test_words_normalizes_like_the_metric():
    assert words("We're  still, still GATHERING.") == ["we're", "still", "gathering"]
    assert words(None) == []
