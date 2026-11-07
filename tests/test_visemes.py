"""visemes-v1: text -> viseme codes (CMUdict) and weighted Levenshtein distance.

Every expected sequence below is derived by hand: the CMUdict 1.1.3 first pronunciation is written in the
comment, stress digits dropped, each phoneme mapped with the ARPAbet column of the visemes-v1 table in
docs/uncertainty-design.md, then consecutive duplicate codes collapsed.
"""

import json
import random
from pathlib import Path

import cmudict
import pytest

from visemes import ALPHABET, ALPHABET_VERSION, load_costs, to_visemes, viseme_distance

DEFAULT_COSTS = {"default_sub": 1, "ins": 1, "del": 1, "sub": {}}


def test_alphabet_is_visemes_v1_in_table_order():
    assert ALPHABET_VERSION == "visemes-v1"
    assert ALPHABET == ["P", "F", "TH", "W", "R", "CH", "S", "T", "K", "A", "E", "O", "REST"]


# One word per row; together they cover all 39 CMUdict phonemes at least once.
@pytest.mark.parametrize(
    "word, expected",
    [
        pytest.param("pat", ["P", "A", "T"], id="P,AE-pat"),  # P AE1 T
        pytest.param("bat", ["P", "A", "T"], id="B-bat"),  # B AE1 T
        pytest.param("mat", ["P", "A", "T"], id="M-mat"),  # M AE1 T
        pytest.param("five", ["F", "A", "F"], id="F,AY,V-five"),  # F AY1 V
        pytest.param("vast", ["F", "A", "S", "T"], id="V,S-vast"),  # V AE1 S T
        pytest.param("thing", ["TH", "E", "K"], id="TH,IH,NG-thing"),  # TH IH1 NG
        pytest.param("this", ["TH", "E", "S"], id="DH-this"),  # DH IH1 S
        pytest.param("we", ["W", "E"], id="W,IY-we"),  # W IY1
        pytest.param("food", ["F", "W", "T"], id="UW,D-food"),  # F UW1 D
        pytest.param("book", ["P", "W", "K"], id="UH,K-book"),  # B UH1 K
        pytest.param("ray", ["R", "E"], id="R,EY-ray"),  # R EY1
        pytest.param("were", ["W", "R"], id="ER-were"),  # W ER1
        pytest.param("cheese", ["CH", "E", "S"], id="CH,Z-cheese"),  # CH IY1 Z
        pytest.param("judge", ["CH", "A", "CH"], id="JH,AH-judge"),  # JH AH1 JH
        pytest.param("shoe", ["CH", "W"], id="SH-shoe"),  # SH UW1
        pytest.param("measure", ["P", "E", "CH", "R"], id="ZH,EH-measure"),  # M EH1 ZH ER0
        pytest.param("nut", ["T", "A", "T"], id="N-nut"),  # N AH1 T
        pytest.param("oil", ["O", "T"], id="OY,L-oil"),  # OY1 L
        pytest.param("go", ["K", "O"], id="G,OW-go"),  # G OW1
        pytest.param("hat", ["K", "A", "T"], id="HH-hat"),  # HH AE1 T
        pytest.param("sing", ["S", "E", "K"], id="NG-sing"),  # S IH1 NG
        pytest.param("top", ["T", "A", "P"], id="AA-top-first-pron"),  # T AA1 P  (2nd pron T AO1 P is ignored)
        pytest.param("dog", ["T", "O", "K"], id="AO-dog"),  # D AO1 G
        pytest.param("now", ["T", "A"], id="AW-now"),  # N AW1
        pytest.param("yes", ["E", "S"], id="Y-yes"),  # Y EH1 S  (Y, EH both E -> collapsed)
    ],
)
def test_word_maps_through_visemes_v1_table(word, expected):
    assert to_visemes(word) == expected


def test_open_the_terminal():
    # open OW1 P AH0 N         -> O P A T
    # the  DH AH0              -> TH A
    # terminal T ER1 M AH0 N AH0 L -> T R P A T A T
    assert to_visemes("open the terminal") == ["O", "P", "A", "T", "TH", "A", "T", "R", "P", "A", "T", "A", "T"]


def test_homophenes_pat_bat_mat_share_visemes_and_have_zero_distance():
    # pat P AE1 T, bat B AE1 T, mat M AE1 T: P/B/M are one lip closure
    pat, bat, mat = to_visemes("pat"), to_visemes("bat"), to_visemes("mat")
    assert pat == bat == mat == ["P", "A", "T"]
    assert viseme_distance(pat, bat) == 0
    assert viseme_distance(bat, mat) == 0


@pytest.mark.parametrize(
    "word, expected",
    [
        ("thank", ["TH", "A", "K"]),  # TH AE1 NG K  (NG, K both K)
        ("send", ["S", "E", "T"]),  # S EH1 N D  (N, D both T)
        ("don't", ["T", "O", "T"]),  # D OW1 N T  (N, T both T)
    ],
)
def test_duplicates_collapse_within_a_word(word, expected):
    assert to_visemes(word) == expected


@pytest.mark.parametrize(
    "phrase, expected",
    [
        # help HH EH1 L P -> K E T P ; me M IY1 -> P E ; P|P at the boundary collapses
        ("help me", ["K", "E", "T", "P", "E"]),
        # this DH IH1 S -> TH E S ; sun S AH1 N -> S A T ; S|S at the boundary collapses
        ("this sun", ["TH", "E", "S", "A", "T"]),
    ],
)
def test_duplicates_collapse_across_word_boundary(phrase, expected):
    assert to_visemes(phrase) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # fallback: th->TH w->W o->O m->P p->P (P P collapsed)
        ("thwomp", ["TH", "W", "O", "P"]),
        # fallback: x->K S, o->O, ch->CH, i->E, t->T, l->T (T T collapsed)
        ("xochitl", ["K", "S", "O", "CH", "E", "T"]),
        # call K AO1 L -> K O T (CMUdict) + xochitl (fallback); T|K boundary, no collapse
        ("call xochitl", ["K", "O", "T", "K", "S", "O", "CH", "E", "T"]),
    ],
)
def test_oov_word_uses_deterministic_letter_fallback(text, expected):
    lexicon = cmudict.dict()
    assert "thwomp" not in lexicon and "xochitl" not in lexicon  # precondition: really out of vocabulary
    assert to_visemes(text) == expected
    assert to_visemes(text) == to_visemes(text)


def test_case_and_punctuation_are_normalized():
    assert to_visemes("Open the TERMINAL!") == to_visemes("open the terminal")
    # hello HH AH0 L OW1 -> K A T O ; world W ER1 L D -> W R T
    assert to_visemes("'Hello,' World...") == ["K", "A", "T", "O", "W", "R", "T"]


def test_apostrophe_inside_word_is_kept():
    # we're W IY1 R -> W E R   (vs. "were" W ER1 -> W R)
    assert to_visemes("We're") == ["W", "E", "R"]
    assert to_visemes("We’re") == ["W", "E", "R"]  # typographic apostrophe from phone keyboards / STT


def test_hyphen_separates_words():
    # well W EH1 L -> W E T ; known N OW1 N -> T O T ; T|T boundary collapses
    assert to_visemes("well-known") == ["W", "E", "T", "O", "T"]


@pytest.mark.parametrize("text", ["", "   ", "?!...", "' '"])
def test_text_without_words_gives_empty_sequence(text):
    assert to_visemes(text) == []


def test_phrase_bank_produces_only_text_codes():
    text_codes = set(ALPHABET) - {"REST"}  # REST is observed-only
    for phrase in Path("data/phrases.txt").read_text().splitlines():
        codes = to_visemes(phrase)
        assert codes, phrase
        assert set(codes) <= text_codes, phrase


# ---- distance -------------------------------------------------------------------------------------------


def test_identical_sequences_have_zero_distance():
    assert viseme_distance(["O", "P", "A", "T"], ["O", "P", "A", "T"]) == 0
    assert viseme_distance([], []) == 0


def test_completely_different_equal_length_sequences_have_distance_one():
    assert viseme_distance(["P", "A", "T"], ["F", "E", "S"]) == 1


def test_one_substitution_in_length_four_is_a_quarter():
    assert viseme_distance(["O", "P", "A", "T"], ["O", "F", "A", "T"]) == pytest.approx(0.25)


def test_insertion_is_normalized_by_the_longer_sequence():
    assert viseme_distance(["P", "A", "T"], ["P", "A", "T", "S"]) == pytest.approx(0.25)
    assert viseme_distance(["P", "A", "T", "S"], ["P", "A", "T"]) == pytest.approx(0.25)
    assert viseme_distance([], ["P", "A"]) == 1


def test_custom_pair_cost_applies_in_both_directions():
    costs = {"default_sub": 1, "ins": 1, "del": 1, "sub": {"P|F": 0.6}}
    assert viseme_distance(["O", "P", "A", "T"], ["O", "F", "A", "T"], costs) == pytest.approx(0.15)
    assert viseme_distance(["O", "F", "A", "T"], ["O", "P", "A", "T"], costs) == pytest.approx(0.15)


def test_explicit_reverse_pair_overrides_direction():
    costs = {"default_sub": 1, "ins": 1, "del": 1, "sub": {"P|F": 0.6, "F|P": 0.2}}
    assert viseme_distance(["O", "P", "A", "T"], ["O", "F", "A", "T"], costs) == pytest.approx(0.15)
    assert viseme_distance(["O", "F", "A", "T"], ["O", "P", "A", "T"], costs) == pytest.approx(0.05)


def test_delete_plus_insert_wins_when_cheaper_than_substitution():
    costs = {"default_sub": 1, "ins": 0.25, "del": 0.25, "sub": {}}
    # P->F by delete P (0.25) + insert F (0.25) = 0.5, over length 4
    assert viseme_distance(["O", "P", "A", "T"], ["O", "F", "A", "T"], costs) == pytest.approx(0.125)


def _random_pairs(n=300, seed=7):
    rng = random.Random(seed)
    codes = ALPHABET
    for _ in range(n):
        yield (
            [rng.choice(codes) for _ in range(rng.randint(0, 8))],
            [rng.choice(codes) for _ in range(rng.randint(0, 8))],
        )


@pytest.mark.parametrize(
    "costs",
    [
        DEFAULT_COSTS,
        {"default_sub": 0.9, "ins": 0.4, "del": 0.4, "sub": {"P|F": 0.3, "T|S": 0.5, "A|E": 0.2, "E|A": 0.2}},
    ],
    ids=["default", "symmetric-custom"],
)
def test_distance_is_symmetric_and_bounded_for_symmetric_costs(costs):
    for a, b in _random_pairs():
        d = viseme_distance(a, b, costs)
        assert 0 <= d <= 1, (a, b, d)
        assert d == pytest.approx(viseme_distance(b, a, costs)), (a, b)


# ---- costs file -----------------------------------------------------------------------------------------


def test_shipped_costs_file_holds_the_defaults():
    costs = load_costs()
    assert costs["default_sub"] == 1
    assert costs["ins"] == 1
    assert costs["del"] == 1
    assert costs["sub"] == {}


def test_distance_without_costs_uses_the_costs_file():
    a, b = to_visemes("open the terminal"), to_visemes("open the turbine")
    assert viseme_distance(a, b) == viseme_distance(a, b, load_costs())


def test_load_costs_reads_an_edited_file(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"alphabet": "visemes-v1", "default_sub": 1, "ins": 1, "del": 1, "sub": {"P|F": 0.6}}))
    costs = load_costs(path)
    assert viseme_distance(["O", "P", "A", "T"], ["O", "F", "A", "T"], costs) == pytest.approx(0.15)


@pytest.mark.parametrize(
    "bad",
    [
        {"alphabet": "visemes-v1", "default_sub": 1, "ins": 1, "del": 1, "sub": {"P|FF": 0.6}},  # unknown code
        {"alphabet": "visemes-v1", "default_sub": 1, "ins": 1.5, "del": 1, "sub": {}},  # cost outside [0, 1]
        {"alphabet": "visemes-v0", "default_sub": 1, "ins": 1, "del": 1, "sub": {}},  # other alphabet
    ],
    ids=["unknown-code", "out-of-range", "wrong-alphabet"],
)
def test_load_costs_rejects_invalid_files(tmp_path, bad):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_costs(path)
