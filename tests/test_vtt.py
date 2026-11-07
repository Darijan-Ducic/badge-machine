"""Lip mode at the clips layer: a muted recording is read with K=3, stored unlabeled, scored by a model trained on the
user's earlier labeled clips only, and labeled later by the user. Every store test runs on both backends; the
readers are fakes, no network."""

import time

import pytest

from clips import NONE_TEXT, backend, clip_detail, label_clip, load, read_clip, record_clip, vtt_clip
from tests.fakes import PHRASES, Gemini, SampleOpen, ScoreClosed, readers, v1_output

# "yes" wins; closed scores keep the runners-up apart: yes .80, hello .08, no .04, turn on .03, ∅ .03, turn off .03
SPREAD = {"closed": {"no": 0.5, "turn on the lights": 0.3}, "gemini": "yes"}
NONE_WIN = dict(sample_open=SampleOpen(v1_output(present=False)), score_closed=ScoreClosed(none=0.95, default=0.01),
                gemini=Gemini(probs=dict.fromkeys(PHRASES, 0.25)))


def read_right(said: str) -> dict:
    return readers(said, gemini=said)


def vtt(store, recording, user="ann", fakes=None):
    return vtt_clip(recording, PHRASES, store, user_id=user, **(fakes or read_right("yes")))


def add_labeled(store, recording, n: int, user="ann"):
    """n STT-labeled clips alternating "yes" / "no", each read right: 10 move a model past its defaults stage."""
    for i in range(n):
        said = PHRASES[i % 2]
        record_clip(recording, phrases=PHRASES, user_id=user, mode="stt_on", k=3, transcript=said, store=store,
                    **read_right(said))


def real_pool(store, clip_id) -> list[dict]:
    """The clip's candidates as GET /clips/<id> scores them (earlier clips only), most probable first, ∅ dropped."""
    return [{"text": c["text"], "p": c["p"]} for c in clip_detail(store, clip_id)["pool"] if c["text"] != NONE_TEXT]


class LandsAfter:
    """A store where `later` (record, video) pairs land right after the next save, as clips from concurrent requests
    can between a lip clip's save and its read-back."""

    def __init__(self, store, later):
        self.inner, self.later = backend(store), later

    def save(self, record, video):
        self.inner.save(record, video)
        for later_record, later_video in self.later:
            self.inner.save(later_record, later_video)
        self.later = []

    def __getattr__(self, name):
        return getattr(self.inner, name)


# ---------------------------------------------------------------- reading a lip clip


def test_a_lip_clip_is_read_with_k3_and_stored_unlabeled_in_lip_mode(store, recording):
    fakes = read_right("yes")

    result = vtt(store, recording, fakes=fakes)

    assert [call[1] for call in fakes["sample_open"].calls] == [3]
    (stored,) = load(store)
    assert (stored["id"], stored["user_id"], stored["mode"], stored["config"]["k"]) == (result["clip"], "ann", "lip", 3)
    assert stored["label"] == {"stt_text": None, "user_edit": None}


def test_vtt_is_the_most_probable_candidate_and_alternatives_are_the_top_three_best_first(store, recording):
    result = vtt(store, recording, fakes=readers("yes", **SPREAD))

    pool = real_pool(store, result["clip"])
    assert [a["text"] for a in result["alternatives"]] == ["yes", "hello", "no"] == [c["text"] for c in pool[:3]]
    assert [a["p"] for a in result["alternatives"]] == pytest.approx([c["p"] for c in pool[:3]])
    assert result["vtt"] == result["alternatives"][0]
    assert result["action"] == "confirm", "a model with no labeled clips behind it never auto-sends"


def test_latency_covers_the_whole_read(store, recording):
    def slow_gemini(clip, phrases, usage=None):
        time.sleep(0.3)
        return Gemini("yes")(clip, phrases, usage)

    result = vtt(store, recording, fakes={**read_right("yes"), "gemini": slow_gemini})

    assert isinstance(result["latency_ms"], int) and result["latency_ms"] >= 300


# ---------------------------------------------------------------- ∅ is never sent


def test_a_none_win_is_not_sent(store, recording):
    result = vtt(store, recording, fakes=NONE_WIN)

    assert clip_detail(store, result["clip"])["pool"][0]["text"] == NONE_TEXT, "∅ is the top pick"
    assert result["action"] == "none"


def test_a_none_win_offers_the_best_real_candidates_never_none(store, recording):
    result = vtt(store, recording, fakes=NONE_WIN)

    assert "∅" not in result["vtt"]["text"]
    assert [a["text"] for a in result["alternatives"]] == [c["text"] for c in real_pool(store, result["clip"])[:3]]
    assert all("∅" not in a["text"] for a in result["alternatives"]) and len(result["alternatives"]) == 3
    assert result["vtt"] == result["alternatives"][0]


# ---------------------------------------------------------------- the model: this user's earlier labeled clips only


def test_earlier_labeled_clips_train_the_model_that_scores_a_lip_clip(store, recording):
    add_labeled(store, recording, 10)

    result = vtt(store, recording)

    assert (result["vtt"]["text"], result["action"]) == ("yes", "auto")


def test_other_users_clips_do_not_train_the_model(store, recording):
    add_labeled(store, recording, 10, user="bob")

    result = vtt(store, recording, user="ann")

    assert result["action"] == "confirm"


def test_clips_stored_after_a_lip_clip_do_not_train_its_model(store, recording):
    later = [read_clip(recording, phrases=PHRASES, user_id="ann", mode="stt_on", k=3, transcript=PHRASES[i % 2],
                       **read_right(PHRASES[i % 2])) for i in range(10)]

    result = vtt(LandsAfter(store, later), recording)

    assert len(load(store, "ann")) == 11, "the later clips did land"
    assert result["action"] == "confirm"
    earlier_only = real_pool(store, result["clip"])[:3]
    assert [a["p"] for a in result["alternatives"]] == pytest.approx([c["p"] for c in earlier_only])


# ---------------------------------------------------------------- the user's label


def test_a_users_label_is_stored_on_that_clip_only(store, recording):
    first, second = vtt(store, recording)["clip"], vtt(store, recording)["clip"]

    assert label_clip(store, first, "turn on the lights", "edit") is True

    assert backend(store).get(first)["label"] == {"stt_text": None, "user_edit": "turn on the lights", "source": "edit"}
    assert backend(store).get(second)["label"] == {"stt_text": None, "user_edit": None}


def test_labeling_an_unknown_clip_changes_nothing(store, recording):
    vtt(store, recording)
    before = load(store)

    assert label_clip(store, "no-such-clip", "yes", "confirm") is False
    assert load(store) == before


@pytest.mark.parametrize("source", ["typed", "", None, "Edit"])
def test_a_label_source_outside_undo_confirm_edit_is_rejected(store, recording, source):
    clip = vtt(store, recording)["clip"]

    with pytest.raises(ValueError):
        label_clip(store, clip, "yes", source)
    assert backend(store).get(clip)["label"] == {"stt_text": None, "user_edit": None}


@pytest.mark.parametrize("text", [None, "", "   ", 3])
def test_a_label_without_text_is_rejected(store, recording, text):
    clip = vtt(store, recording)["clip"]

    with pytest.raises(ValueError):
        label_clip(store, clip, text, "edit")
    assert backend(store).get(clip)["label"] == {"stt_text": None, "user_edit": None}


def test_labeled_lip_clips_train_later_reads(store, recording):
    for i in range(10):
        said = PHRASES[i % 2]
        clip = vtt(store, recording, fakes=read_right(said))["clip"]
        label_clip(store, clip, said, ("undo", "confirm", "edit")[i % 3])

    result = vtt(store, recording)

    assert (result["vtt"]["text"], result["action"]) == ("yes", "auto")
