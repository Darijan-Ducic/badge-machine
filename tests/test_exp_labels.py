"""Label-quality instruments: STT text aligned into the official transcript, and the metrics read off the alignment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from exp_labels import align, boundary, doubles, press_text, words  # noqa: E402

REF_TEXT = ("We are still gathering all the facts about what happened, but we do know that the police have one "
            "suspect in custody. And the federal government stands ready to do everything necessary.")



def ref():
    return words(REF_TEXT)


def test_words_spells_numbers_and_drops_punctuation():
    assert words("At least 12 people — in Aurora, Colorado.") == ["at", "least", "twelve", "people", "in", "aurora",
                                                                 "colorado"]


def test_words_keeps_contractions_with_curly_apostrophes():
    assert words("children’s hospital, who’s") == ["children's", "hospital", "who's"]


def test_words_spells_compound_numbers():
    assert words("2,500 and 21") == ["two", "thousand", "five", "hundred", "and", "twenty", "one"]


def test_align_finds_the_span_inside_a_longer_reference():
    hyp = words("but we do know that the police have one suspect")
    a = align(hyp, ref())
    assert (ref()[a.start:a.end], a.edits) == (hyp, 0)


def test_align_counts_a_substitution_and_an_insertion():
    a = align(words("we do knew that the police police have"), ref())
    assert a.edits == 2 and ref()[a.start:a.end] == words("we do know that the police have")


def test_align_counts_words_missing_inside_the_span():
    a = align(words("the federal government ready to do everything"), ref())
    assert a.edits == 1 and ref()[a.start:a.end] == words("the federal government stands ready to do everything")


def test_doubles_counts_a_repeat_the_reference_does_not_have():
    hyp = words("that the police police have one suspect")
    assert doubles(hyp, align(hyp, ref())) == 1


def test_doubles_ignores_a_repeat_that_was_spoken():
    ref = words("I know that that is hard for all of us")
    hyp = words("I know that that is hard")
    assert doubles(hyp, align(hyp, ref)) == 0


def test_boundary_counts_reference_words_between_two_turns_as_lost():
    first, second = align(words("we are still gathering all the facts"), ref()), align(words("what happened but we"), ref())
    assert boundary(first, second) == {"lost": 1, "overlap": 0}  # "about" is in neither turn


def test_boundary_counts_a_word_heard_by_both_turns_as_overlap():
    first = align(words("we are still gathering all the facts about"), ref())
    second = align(words("about what happened"), ref())
    assert boundary(first, second) == {"lost": 0, "overlap": 1}


def test_press_text_keeps_the_address_paragraphs_only():
    html = ('<div class="field field-name-field-transcript"><div class="field-items"><div class="field-item even">'
            '<p class="rtecenter"><strong>Remarks of President Barack Obama</strong><br/><strong>Weekly Address'
            '</strong></p><p>\n\tHi, everybody.  It&#8217;s a <em>good</em> day.</p><p>\n\tThank you.</p><p>\n\t###</p>'
            '</div></div></div><div class="footer"><p>Share This</p></div>')
    assert press_text(html) == "Hi, everybody. It’s a good day. Thank you."
