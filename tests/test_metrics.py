"""metrics.py: chance-aware VTT metrics and the power calculation. Expected values are computed by hand in comments."""

import math

import pytest

import metrics


# ---------------------------------------------------------------- phoneme accuracy (CMUdict)


def test_phoneme_acc_counts_one_substitution_in_three_phonemes():
    # bat = B AE T, cat = K AE T: one substitution over 3 STT phonemes
    assert metrics.phoneme_acc(["bat"], ["cat"]) == pytest.approx(2 / 3)


def test_phoneme_acc_pools_edits_over_turns():
    # (bat, cat) 1 edit / 3, (cat, cat) 0 / 3 -> 1 - 1/6
    assert metrics.phoneme_acc(["bat", "cat"], ["cat", "cat"]) == pytest.approx(5 / 6)


def test_phoneme_acc_separates_what_visemes_merge():
    # b and p are one viseme (P) but two phonemes: viseme acc is 1, phoneme acc is 2/3
    assert metrics.phoneme_acc(["bat"], ["pat"]) == pytest.approx(2 / 3)


def test_phoneme_acc_out_of_vocabulary_word_matches_only_itself():
    assert metrics.phoneme_acc(["zxqv"], ["zxqv"]) == 1.0
    assert metrics.phoneme_acc(["blorpf"], ["zxqv"]) == 0.0  # one OOV token each: one substitution


def test_phoneme_acc_is_none_without_stt_phonemes():
    assert metrics.phoneme_acc(["cat"], [""]) is None


# ---------------------------------------------------------------- length-normalized viseme accuracy


def test_viseme_acc_norm_divides_by_the_longer_sequence():
    # "bob bob" -> P A P A P (5, duplicates collapse), "bob" -> P A P (3): 2 insertions.
    # experiment.viseme_acc = 1 - 2/3; normalized by max length = 1 - 2/5
    assert metrics.viseme_acc_norm(["bob bob"], ["bob"]) == pytest.approx(3 / 5)


def test_viseme_acc_norm_is_never_negative_for_long_reads():
    # 7 words of "bob" against one: viseme_acc would be far below 0
    assert metrics.viseme_acc_norm(["bob top bob top bob top bob"], ["bob"]) >= 0


# ---------------------------------------------------------------- content words


def test_content_recall_ignores_stopwords():
    # STT content words: cat, sat, mat (the, on, a are stopwords); the read has cat and mat
    assert metrics.content_recall(["a cat on the mat"], ["the cat sat on a mat"]) == pytest.approx(2 / 3)


def test_content_recall_clips_repeated_words():
    # STT content counts {cat: 2, dog: 1}; one "cat" in the read recovers 1 of 3
    assert metrics.content_recall(["cat"], ["cat dog cat"]) == pytest.approx(1 / 3)


def test_content_recall_pools_over_turns_and_skips_turns_without_content():
    # turn 1: 1 of 1 content word; turn 2: STT "the a" has none, contributes nothing; turn 3: 0 of 2
    assert metrics.content_recall(["cat", "x", "y"], ["cat", "the a", "dog mat"]) == pytest.approx(1 / 3)


def test_keyword_hit_is_the_share_of_turns_with_any_content_word():
    assert metrics.keyword_hit(["cat", "dog"], ["the cat sat", "the cat sat"]) == 0.5


def test_keyword_hit_stopword_overlap_is_not_a_hit():
    assert metrics.keyword_hit(["the"], ["the cat"]) == 0.0


# ---------------------------------------------------------------- impostor-corrected viseme accuracy


def test_viseme_lift_is_own_minus_other_turns():
    # visemes: bob = P A P, top = T A P. acc(bob, bob) = 1, acc(bob, top) = 2/3 -> lift 1/3 per turn
    assert metrics.viseme_lift(["bob", "top"], ["bob", "top"]) == pytest.approx(1 / 3)


def test_viseme_lift_of_a_constant_read_is_zero():
    stts = ["bob", "top", "the cat sat on a mat", "we need to keep going"]
    for constant in ("bob", "the to of in we and a that our at for this", "we need to keep going together"):
        assert metrics.viseme_lift([constant] * len(stts), stts) == pytest.approx(0.0, abs=1e-12)


def test_viseme_lift_turns_returns_one_value_per_turn():
    # constant "bob": turn bob = 1 - 2/3, turn top = 2/3 - 1
    assert metrics.viseme_lift_turns(["bob", "bob"], ["bob", "top"]) == pytest.approx([1 / 3, -1 / 3])


# ---------------------------------------------------------------- closed-set sentence identification


def test_sentence_id_finds_each_turns_own_stt():
    out = metrics.sentence_id(["bob", "top"], ["bob", "top"])
    assert out == {"top1": 1.0, "percentile": 1.0, "n": 2}


def test_sentence_id_constant_read_scores_chance_exactly():
    stts = ["bob", "top", "the cat sat on a mat", "we need to keep going"]
    out = metrics.sentence_id(["the cat sat"] * 4, stts)
    assert out["top1"] == pytest.approx(1 / 4)
    assert out["percentile"] == pytest.approx(0.5)


def test_sentence_id_ties_share_credit():
    # identical STT texts: the read is equally near both, so each turn gets half a top-1 and a 0.5 percentile
    out = metrics.sentence_id(["bob", "bob"], ["bob", "bob"])
    assert out["top1"] == pytest.approx(0.5)
    assert out["percentile"] == pytest.approx(0.5)


def test_sentence_id_turns_gives_rank_percentiles():
    # read "bob": distance to bob 0, to top 1/3 -> own STT beats 1 of 1 impostor; read "top" for STT bob: loses
    assert metrics.sentence_id_turns(["bob", "bob"], ["bob", "top"]) == pytest.approx([1.0, 0.0])


# ---------------------------------------------------------------- power


def test_paired_power_matches_the_t_test_table():
    # one-sample / paired t, two-sided alpha .05: effect = 1 SD at n = 10 has power 0.803 (G*Power)
    assert metrics.paired_power(effect=1.0, sd=1.0, n=10) == pytest.approx(0.803, abs=0.002)


def test_turns_needed_uses_t_not_normal():
    # half an SD at 80% power: 34 pairs by the t test (the normal approximation would say 32)
    assert metrics.turns_needed(effect=0.5, sd=1.0) == 34


def test_mde_is_the_effect_with_exactly_the_target_power():
    effect = metrics.mde_paired(sd=0.05, n=40)
    assert metrics.paired_power(effect=effect, sd=0.05, n=40) == pytest.approx(0.8, abs=1e-6)
    assert metrics.mde_paired(sd=0.05, n=40) < metrics.mde_paired(sd=0.05, n=20)


def test_sign_power_six_pairs_rejects_only_a_clean_sweep():
    # m = 6: P(6 or 0 ups | fair) = 2/64 < .05 rejects; 5 of 6 gives 14/64 > .05. Power = .9^6 + .1^6
    assert metrics.sign_power(6, 0.9) == pytest.approx(0.9**6 + 0.1**6)


def test_sign_power_five_pairs_can_never_reject():
    # m = 5: the most lopsided split has p = 2/32 = .0625 > .05
    assert metrics.sign_power(5, 0.99) == 0.0


def test_sign_turns_needed_is_the_smallest_powered_count():
    m = metrics.sign_turns_needed(0.8)
    assert metrics.sign_power(m, 0.8) >= 0.8
    assert all(metrics.sign_power(k, 0.8) < 0.8 for k in range(1, m))


# ---------------------------------------------------------------- sample-to-sample noise


def test_split_half_sd_from_the_stored_samples():
    # turn 1, K = 2: picks "bob" and "top" against STT bob -> differences +-1/3; turn 2: identical samples -> 0.
    # sqrt(mean(1/9, 0)) = sqrt(1/18)
    sd = metrics.split_half_sd([["bob", "top"], ["cat", "cat"]], ["bob", "cat"], metrics.per_turn_viseme_acc)
    assert sd == pytest.approx(math.sqrt(1 / 18))


def test_split_half_sd_compares_the_picks_of_each_half():
    # K = 4, samples [x, x, y, y] for STT x: splitting 2|2, pairs {x,x}|{y,y} and {y,y}|{x,x} differ, {x,y}|{x,y}
    # picks the first of each half: x|x (from [x,y] and [x,y]) -> 0. 6 splits: 2 of them differ by 1 (word acc 1
    # vs 0, "dog" for "cat"), so mean square = 2/6
    sd = metrics.split_half_sd([["cat", "cat", "dog", "dog"]], ["cat"], metrics.per_turn_word_acc)
    assert sd == pytest.approx(math.sqrt(2 / 6))
